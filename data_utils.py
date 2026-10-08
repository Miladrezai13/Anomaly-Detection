"""Dataset preparation and calibration helpers; no model dependencies."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import tarfile
import zipfile
import numpy as np

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

def find_dataset_root(folder):
    """Find a unique category parent up to four folder levels below folder."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    candidates = []
    for current, dirs, _ in os.walk(folder):
        current = Path(current)
        depth = len(current.relative_to(folder).parts)
        if depth >= 4:
            dirs[:] = []
        if any((current / cat / "train" / "good").is_dir() for cat in CATEGORIES):
            candidates.append(current)
            dirs[:] = []
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one dataset root, found {len(candidates)}: {candidates}. "
            "Set DATA_SOURCE to the exact extracted dataset folder. "
            "If it is still compressed, set DATA_SOURCE to the archive file."
        )
    return candidates[0]

def validate_categories(root, categories=CATEGORIES):
    root = Path(root)
    missing = [cat for cat in categories if not (root / cat / "train" / "good").is_dir()]
    if missing:
        raise ValueError(f"Missing categories: {missing}. Expected category/train/good.")
    counts = {cat: len(image_paths(root / cat / "train" / "good")) for cat in categories}
    if any(n < 10 for n in counts.values()):
        raise ValueError(f"Too few normal images for fitting and calibration: {counts}")
    return counts

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def safe_extract(archive_path, destination):
    """Extract ZIP or TAR into destination, rejecting escaped paths and ZIP links."""
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"Unsafe archive path: {item.filename}")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f"ZIP symbolic link not supported: {item.filename}")
            archive.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError("Use Python 3.12+ for filtered TAR extraction.")
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination, filter="data")
    else:
        raise ValueError("Provide an extracted folder or a ZIP/TAR archive.")

def prepare_source(source, workspace):
    """Use an extracted Drive folder or copy/extract one specified archive locally."""
    source, workspace = Path(source), Path(workspace)
    if source.is_dir():
        return find_dataset_root(source)
    if not source.is_file():
        raise FileNotFoundError(f"DATA_SOURCE does not exist: {source}")
    signature = {"source": str(source.resolve()), "bytes": source.stat().st_size,
                 "mtime_ns": source.stat().st_mtime_ns}
    destination = workspace / "archive_data"
    marker = workspace / "archive_ready.json"
    if marker.exists() and json.loads(marker.read_text()) == signature:
        return find_dataset_root(destination)
    if destination.exists():
        shutil.rmtree(destination)
    if marker.exists():
        marker.unlink()
    workspace.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(workspace).free < source.stat().st_size * 3:
        raise OSError("Insufficient runtime disk space for archive copy and extraction.")
    local_archive = workspace / source.name
    shutil.copy2(source, local_archive)
    print("Extracting archive into runtime storage; this can take several minutes.")
    safe_extract(local_archive, destination)
    root = find_dataset_root(destination)
    atomic_json(marker, signature)
    local_archive.unlink()
    return root

def stage_category(source_root, category, workspace):
    """Copy only one category at a time; skip unrelated notebooks and output files."""
    source = Path(source_root) / category
    local_parent = Path(workspace) / "active_category"
    local_parent.mkdir(parents=True, exist_ok=True)
    destination = local_parent / category
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    for role in ("train", "test", "ground_truth"):
        if (source / role).is_dir():
            shutil.copytree(source / role, destination / role)
    return destination

def make_normal_manifest(category_root, seed=42):
    """Disjoint 60/20/20 groups from official train/good; hash contents for resume."""
    category_root = Path(category_root)
    paths = image_paths(category_root / "train" / "good")
    if len(paths) < 10:
        raise ValueError("Need at least ten normal training images.")
    order = np.random.default_rng(seed).permutation(len(paths))
    n_fit, n_dev = round(.60 * len(paths)), round(.20 * len(paths))
    rows = []
    for rank, index in enumerate(order):
        path = paths[index]
        split = "fit" if rank < n_fit else ("dev" if rank < n_fit + n_dev else "calibration")
        rows.append({"path": str(path.relative_to(category_root)), "split": split,
                     "sha256": sha256_file(path)})
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return rows, fingerprint

def calibration_record(scores, alpha=.05):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Calibration needs at least two finite image scores.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    threshold = float(np.quantile(scores, 1 - alpha, method="higher"))
    return {"threshold": threshold, "target_fpr": float(alpha), "comparison": ">",
            "count": len(scores), "observed_calibration_fpr": float((scores > threshold).mean()),
            "map_display_max": None}

