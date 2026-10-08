# MVTec AD anomaly detection

Train a PatchCore model for each MVTec AD product category and inspect images with a Gradio app. The app shows an anomaly score, a heatmap, and regions that may need review.

The default model uses Anomalib 2.3.0 with a pretrained ResNet18 backbone, 256 × 256 images, and a 1% coreset. Each category has its own model and threshold.

## Run locally

Use **Python 3.11 or 3.12**. Clone this repository, open a terminal in the project folder, and create an environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\activate` instead. PyTorch and torchvision are installed together through the requirements file. For CUDA, use the compatible pair from the [PyTorch installer](https://pytorch.org/get-started/locally/).

### Get the models

**Trained checkpoints are not included, and no model download is currently provided.** Train them with the notebook below, or copy a trained run folder from another machine. Installing the packages alone is not enough to run the UI.

Place the run folder in this layout:

```text
mvtec-all-categories/
├── app.py
├── data_utils.py
├── mvtec_patchcore.py
├── requirements.txt
└── runs/
    └── patchcore_resnet18_256_r001_seed42/
        └── models/
            ├── bottle.pt
            └── ...
```

At least one category checkpoint is required. These must be checkpoints created by this project's training code; arbitrary PatchCore weights are not interchangeable.

```bash
python app.py --run-dir runs/patchcore_resnet18_256_r001_seed42
```

Open **http://127.0.0.1:7860**, choose the product category, upload an image, and click **Inspect image**. The dataset is not needed for inference. The app uses CPU when CUDA is unavailable.

For a downloaded run bundle containing `source/` and `models/`, install the requirements from `source/` and run `python app.py --run-dir ..` there.

## Train models

Open `mvtec_all_categories_train_and_ui.ipynb` in Google Colab with a GPU runtime, or in local Jupyter using the same Python environment. For local Jupyter, install `jupyterlab` and `ipykernel` in that environment and launch `python -m jupyter lab`.

1. Download and extract the [original MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
2. In section 2, set `PROJECT_DIR` and `DATA_SOURCE`. Local paths default to the notebook's working directory. In Colab, models are saved to Google Drive.
3. Run the cells in order. Section 5 checks that all selected categories contain training images.
4. Section 6 trains and reloads the first category. Section 7 trains the remaining categories.
5. Use section 8 to inspect an image, or section 10 to export a standalone app bundle.

The dataset should contain paths such as `bottle/train/good/000.png`. `DATA_SOURCE` can also point to a ZIP or TAR archive.

Normal training images are split into fitting, development, and calibration groups (60/20/20). Official test images are used only for the optional final evaluation. Completed categories can be resumed when their configuration and training data match. Use a new run folder when changing model settings.

All-category training can take substantial time, especially on CPU. Saved checkpoints include the backbone and memory bank, so allow space for the model files.

## Troubleshooting

- **Missing package:** activate `.venv` and install `requirements.txt` using that environment's Python.
- **No trained category models:** check `--run-dir`. It must point to the folder containing `models/`, not to `models/` itself.
- **Too few normal images:** check the dataset extraction and `DATA_SOURCE`. Every selected category needs at least ten images in `train/good`.
- **Port already in use:** run with `--port 7861` and open the matching address.

## Results and limitations

No benchmark results are included yet. Local checks loaded the saved bottle model, inspected one training image, and built the Gradio interface. The notebook has not been rerun end to end, and browser interaction still needs verification.

Scores are not defect probabilities. The app does not identify defect types or select the product category automatically. Different lighting, backgrounds, or viewpoints can affect the result. The default configuration is not a reproduction of the PatchCore paper's benchmark settings.

The optional evaluation saves image AUROC, recall, false-positive rate, error counts, and individual scores. It does not compute pixel-level metrics.

## GitHub and hosting

This repository contains the code and notebook. Dataset images, trained models, temporary working data, and virtual environments are excluded by `.gitignore`. Notebook outputs are cleared before publishing.

GitHub does not host this Python app through GitHub Pages. A hosted demo is planned for Hugging Face Spaces; deployment configuration and a model download will be added when it is ready.

## References

- [MVTec AD dataset and licence](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [PatchCore paper](https://arxiv.org/abs/2106.08265)
- [Anomalib PatchCore implementation, v2.3.0](https://github.com/open-edge-platform/anomalib/blob/v2.3.0/src/anomalib/models/image/patchcore/torch_model.py)

Follow the MVTec dataset's published licence terms when using or distributing its data.
