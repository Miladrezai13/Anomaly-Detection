"""Anomalib 2.3.0 PatchCore with explicit normal-only splits and raw scoring."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from data_utils import (CATEGORIES, image_paths, atomic_json, make_normal_manifest,
                        calibration_record, stage_category, validate_categories)

EXPECTED_ANOMALIB = "2.3.0"
FORMAT_VERSION = 1
DEFAULT_CONFIG = {"backbone": "resnet18", "layers": ["layer2", "layer3"],
                  "image_size": 256, "coreset_ratio": .01, "num_neighbors": 9,
                  "batch_size": 4, "seed": 42, "target_fpr": .05,
                  "preprocess": "RGB_full_resize_bilinear_ImageNet_normalization_v1"}

def check_version():
    version = importlib.metadata.version("anomalib")
    if version != EXPECTED_ANOMALIB:
        raise RuntimeError(f"Use anomalib=={EXPECTED_ANOMALIB}; installed: {version}")

def environment():
    return {"torch": str(torch.__version__), "anomalib": EXPECTED_ANOMALIB,
            "numpy": str(np.__version__), "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            "saved_utc": datetime.now(timezone.utc).isoformat()}

def make_transform(config):
    return transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"]),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])

def decode_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()

class ImageDataset(Dataset):
    def __init__(self, paths, config):
        self.paths = [Path(p) for p in paths]
        self.transform = make_transform(config)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        return self.transform(decode_image(self.paths[index]))

def loader(paths, config, batch_size=None):
    return DataLoader(ImageDataset(paths, config), shuffle=False, num_workers=0,
                      batch_size=batch_size or config["batch_size"],
                      pin_memory=torch.cuda.is_available())

def new_model(config, pretrained=True):
    check_version()
    return PatchcoreModel(layers=config["layers"], backbone=config["backbone"],
                         pre_trained=pretrained, num_neighbors=config["num_neighbors"])

@torch.no_grad()
def score_paths(model, paths, config, device, desc="Scoring"):
    scores, map_peaks = [], []
    model.eval()
    for images in tqdm(loader(paths, config, batch_size=1), desc=desc):
        prediction = model(images.to(device))
        scores.extend(prediction.pred_score.detach().cpu().reshape(-1).tolist())
        peaks = prediction.anomaly_map.flatten(1).amax(1)
        map_peaks.extend(peaks.detach().cpu().tolist())
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite model scores; check data and memory bank.")
    return values, np.asarray(map_peaks, dtype=float)

def write_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def read_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("Unsupported checkpoint format.")
    if payload.get("anomalib_version") != EXPECTED_ANOMALIB:
        raise RuntimeError("Checkpoint/library version mismatch.")
    bank = payload["state_dict"]["memory_bank"]
    if bank.ndim != 2 or bank.shape[0] < 1 or not torch.isfinite(bank).all():
        raise RuntimeError("Checkpoint contains an invalid memory bank.")
    if not np.isfinite(payload["calibration"]["threshold"]):
        raise RuntimeError("Checkpoint threshold is invalid.")
    return payload

def load_model(path, device):
    payload = read_checkpoint(path)
    if Path(path).stem != payload["category"]:
        raise RuntimeError("Checkpoint filename and saved category do not match.")
    model = new_model(payload["config"], pretrained=False)
    # Make buffer loading explicit; do not depend on automatic buffer resizing.
    model.memory_bank = torch.empty_like(payload["state_dict"]["memory_bank"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload

def train_category(local_category, run_dir, category, config, force=False):
    check_version()
    config = dict(config)
    if config["backbone"] not in {"resnet18", "wide_resnet50_2"}:
        raise ValueError("This preprocessing is supported for the configured ImageNet backbones only.")
    if not 0 < config["coreset_ratio"] <= 1:
        raise ValueError("coreset_ratio must be in (0, 1].")
    local_category, run_dir = Path(local_category), Path(run_dir)
    rows, fingerprint = make_normal_manifest(local_category, config["seed"])
    manifest_path = run_dir / "manifests" / f"{category}.json"
    checkpoint = run_dir / "models" / f"{category}.pt"
    if checkpoint.exists() and not force:
        existing = read_checkpoint(checkpoint)
        if existing["config"] != config or existing["data_fingerprint"] != fingerprint:
            raise RuntimeError(f"{category}: saved configuration/data differ. Use a new RUN_DIR.")
        atomic_json(manifest_path, rows)
        print(f"{category}: verified saved model; skipping fitting.")
        return checkpoint
    atomic_json(manifest_path, rows)
    groups = {name: [local_category / r["path"] for r in rows if r["split"] == name]
              for name in ("fit", "dev", "calibration")}
    assert not set(groups["fit"]) & set(groups["calibration"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model(config).to(device)
    # The outer model must be in training mode to collect patch embeddings.
    # Freeze backbone batch normalization and all other pretrained behaviour.
    model.train()
    model.feature_extractor.eval()
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for images in tqdm(loader(groups["fit"], config), desc=f"{category}: normal features"):
                model(images.to(device))
            print(f"{category}: selecting representative memory features...")
            model.subsample_embedding(sampling_ratio=config["coreset_ratio"])
        model.eval()
        calibration_scores, map_peaks = score_paths(
            model, groups["calibration"], config, device, f"{category}: calibration")
        cal = calibration_record(calibration_scores, config["target_fpr"])
        cal["map_display_max"] = max(float(np.quantile(map_peaks, .99)), 1e-8)
        cal["scores"] = calibration_scores.tolist()
        payload = {
            "format_version": FORMAT_VERSION, "category": category, "config": config,
            "anomalib_version": EXPECTED_ANOMALIB, "data_fingerprint": fingerprint,
            "calibration": cal, "split_counts": {k: len(v) for k, v in groups.items()},
            "memory_vectors": int(model.memory_bank.shape[0]),
            "fit_and_calibration_seconds": float(time.perf_counter() - start),
            "environment": environment(),
            "state_dict": {k: value.detach().cpu() for k, value in model.state_dict().items()},
        }
        write_checkpoint(checkpoint, payload)
        summary = {k: v for k, v in payload.items() if k != "state_dict"}
        atomic_json(run_dir / "calibration" / f"{category}.json", summary)
        print(f"Saved {checkpoint} ({checkpoint.stat().st_size / 1024**2:.1f} MiB)")
        return checkpoint
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train_all(source_root, run_dir, workspace, config=None, categories=CATEGORIES, force=False):
    config = dict(DEFAULT_CONFIG if config is None else config)
    validate_categories(source_root, categories)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("RUN_DIR has another configuration. Choose a new RUN_DIR.")
    atomic_json(config_path, config)
    completed = []
    for category in categories:
        local_category = stage_category(source_root, category, workspace)
        try:
            train_category(local_category, run_dir, category, config, force)
            completed.append(category)
            atomic_json(run_dir / "progress.json", {"completed_in_this_run": completed})
        finally:
            import shutil
            shutil.rmtree(local_category, ignore_errors=True)
    return run_dir

def evaluate_all(source_root, run_dir, workspace, categories=CATEGORIES):
    """Image-level evaluation only; not used to select thresholds or tune settings."""
    from sklearn.metrics import roc_auc_score, confusion_matrix
    import pandas as pd
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for category in categories:
        local = stage_category(source_root, category, workspace)
        model = None
        try:
            model, payload = load_model(run_dir / "models" / f"{category}.pt", device)
            _, fingerprint = make_normal_manifest(local, payload["config"]["seed"])
            if fingerprint != payload["data_fingerprint"]:
                raise RuntimeError("Evaluation source does not match fitted category data.")
            test_dir = local / "test"
            if not test_dir.is_dir():
                raise FileNotFoundError(f"Missing official test directory: {test_dir}")
            records, paths = [], []
            for group in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                for path in image_paths(group):
                    paths.append(path)
                    records.append({"path": str(path.relative_to(local)),
                                    "defect_group": group.name, "label": int(group.name != "good")})
            if not paths:
                raise ValueError(f"No test images for {category}")
            scores, _ = score_paths(model, paths, payload["config"], device, f"{category}: test")
            labels = np.array([r["label"] for r in records])
            predicted = (scores > payload["calibration"]["threshold"]).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
            metric = {"category": category,
                      "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
                      "recall": float(tp / (tp + fn)) if tp + fn else None,
                      "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
                      "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                      "threshold": payload["calibration"]["threshold"]}
            for row, score, decision in zip(records, scores, predicted):
                row.update(score=float(score), decision=int(decision))
            atomic_json(run_dir / "evaluation" / f"{category}_predictions.json", records)
            atomic_json(run_dir / "evaluation" / f"{category}_metrics.json", metric)
            summaries.append(metric)
        finally:
            del model
            import shutil
            shutil.rmtree(local, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    table = pd.DataFrame(summaries)
    table.to_csv(run_dir / "evaluation" / "summary.csv", index=False)
    return table

class Inspector:
    """Keep only the current category on the device, with serialized inference."""
    def __init__(self, run_dir, device=None):
        self.run_dir = Path(run_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.payload = self.category = None
        self.lock = threading.Lock()
    def categories(self):
        return [cat for cat in CATEGORIES if (self.run_dir / "models" / f"{cat}.pt").exists()]
    @torch.no_grad()
    def inspect(self, image, category):
        if not isinstance(image, Image.Image):
            raise ValueError("Upload an image first.")
        if image.width * image.height > 25_000_000:
            raise ValueError("Use an image smaller than 25 megapixels.")
        if category not in self.categories():
            raise ValueError("The selected category has no saved model.")
        with self.lock:
            if self.category != category:
                self.model = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.model, self.payload = load_model(
                    self.run_dir / "models" / f"{category}.pt", self.device)
                self.category = category
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = make_transform(self.payload["config"])(image).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = self.model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            score = float(prediction.pred_score.detach().cpu().reshape(-1)[0])
            raw_map = prediction.anomaly_map.detach().cpu().numpy().reshape(
                self.payload["config"]["image_size"], self.payload["config"]["image_size"])
            if not np.isfinite(score) or not np.isfinite(raw_map).all():
                raise RuntimeError("Invalid model output.")
            # Fixed category display scale comes exclusively from calibration normals.
            scale = max(self.payload["calibration"]["map_display_max"] * 2, 1e-8)
            display_map = np.clip(raw_map / scale, 0, 1)
            from matplotlib import colormaps
            heatmap = Image.fromarray((colormaps["inferno"](display_map)[..., :3] * 255).astype(np.uint8))
            base = image.resize(heatmap.size, Image.Resampling.BILINEAR)
            overlay = Image.blend(base, heatmap, .45)
            threshold = self.payload["calibration"]["threshold"]
            result = {"category": category, "decision": (
                "Review recommended" if score > threshold else "No anomaly detected at this threshold"),
                "anomaly_score": score, "threshold": threshold,
                "scoring_ms_excluding_load_and_preprocessing": round(elapsed, 2),
                "model": "PatchCore / " + self.payload["config"]["backbone"]}
            return base, heatmap, overlay, result
