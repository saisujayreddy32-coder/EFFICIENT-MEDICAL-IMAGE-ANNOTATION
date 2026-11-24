# MedAnnotate AI
## Active Learning for Efficient Medical Image Annotation

A local full-stack research application for chest X-ray lung segmentation. It combines a Flask REST backend, a responsive browser-based annotation workspace, a PyTorch U-Net model, and active-learning sample selection. This release is configured for **Windows 64-bit and Python 3.14.6**.

> **Research and education only.** This software is not a medical device, does not diagnose disease, and must not be used for clinical decisions. All annotations and generated masks require appropriate expert review.

## Application modules
- Overview dashboard for dataset, model and workflow status.
- Dataset Manager for recursive image/mask discovery and pairing.
- Annotation Studio with a browser-based paint/erase mask editor.
- U-Net training with binary cross-entropy plus Dice loss.
- Random-sampling baseline and uncertainty-ranked annotation queue.
- MC-dropout predictive-entropy scoring after a trained model is available.
- Single-image segmentation inference and model-history metrics.

## 1. Dataset
Download and extract the Kaggle dataset: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

The screenshots supplied for this project show this structure:

```text
Lung Segmentation/
  test/
  masks/
  ClinicalReadings/
  CXR_png/
```

The scanner searches recursively. It identifies mask folders by names containing `mask`, `label`, or `ground`, and image folders by names containing `cxr` or `image`. It pairs files using normalized filename stems. Files without matching masks remain in the unlabelled pool. Confirm the scan counts before training.

## 2. Python 3.14.6 installation (Windows)

This project no longer pins NumPy 1.26.x. Python 3.14 requires a newer NumPy wheel; this release uses `numpy>=2.3.4,<2.6`. The Windows setup script installs a binary NumPy wheel and CPU PyTorch, so a local C/C++ compiler or Visual Studio Build Tools is not required for these packages. PyTorch's official Windows instructions list Python 3.10–3.14 as supported.

1. Extract the project ZIP.
2. Open the extracted `MedAnnotateAI` folder.
3. Double-click `setup_windows.bat` and wait for installation to finish.
4. When setup succeeds, double-click `start_windows.bat`.
5. Open http://127.0.0.1:8000 in a browser.

Alternatively, from PowerShell in the project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: "numpy>=2.3.4,<2.6"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run.py
```

The included setup uses CPU PyTorch for broad Windows compatibility. If you have a supported NVIDIA GPU and want CUDA acceleration, use the official PyTorch selector at https://pytorch.org/get-started/locally/ to choose the matching CUDA wheel instead of the CPU wheel.

## 3. Connect your dataset

In Dataset Manager, enter the full path to the `Lung Segmentation` folder, for example:

```text
C:\Users\YourName\Downloads\Lung Segmentation
```

Click **Scan dataset**. Review the number of paired images and unlabelled images. Do not include the `CXR_png` or `masks` folder alone; select their common parent folder.

## 4. Suggested experiment
1. Scan the dataset and inspect the discovered image-mask pairs.
2. Keep a fixed held-out test set separate from active-learning selection.
3. Establish a small initial labelled training set.
4. Train the U-Net and record validation Dice.
5. Compare Random baseline and Uncertainty first at identical annotation budgets.
6. Annotate selected images, retrain, and record results at each cycle.
7. Report Dice, IoU, precision/recall, labelled-image count and annotation savings.

Before a trained model exists, the queue uses a clearly identified image-information proxy (histogram entropy and edge density); it is not model uncertainty. After training, uncertainty ranking uses repeated stochastic predictions with dropout active and ranks by predictive entropy.

The dashboard's pool-coverage estimate is not a measured annotation-saving result. A defensible saving claim requires comparing strategies at equal performance targets and equal test sets.

## 5. Project structure

```text
MedAnnotateAI/
  backend/          Flask API and U-Net/active-learning logic
  frontend/         Responsive HTML, CSS and JavaScript interface
  data/              Local uploads and saved annotations
  models/            Trained model checkpoints
  outputs/           Segmentation outputs
  requirements.txt
  setup_windows.bat  Python 3.14.6 setup
  start_windows.bat  Application launcher
  run.py
```

## 6. Troubleshooting
- If `py -3.14` is not found, install Python 3.14 64-bit and verify with `py --list`.
- If PowerShell blocks activation, use the supplied `.bat` scripts instead.
- If installation is interrupted, rerun `setup_windows.bat`; it recreates the isolated `.venv`.
- If the browser cannot connect, confirm that the terminal still shows the Flask server running at `127.0.0.1:8000`.
- Training on CPU can take time. Start with a small number of epochs.

## Dataset attribution
Review the Kaggle dataset card and accompanying readme documents for required attribution and publication citations before redistributing results or publishing research: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
