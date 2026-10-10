# Gait Abnormality Classification

This project preprocesses Kinect skeleton data and classifies six gait categories:
normal, antalgic, stiff-legged, lurching, steppage, and Trendelenburg.

## Project layout

- `src/build_parquet.py` - converts the downloaded source dataset into balanced
  Parquet files.
- `data/Pathological_Gaits/` - local source dataset directory (not included in
  the repository).
- `data/parquet/` - generated Parquet files used by the notebooks.
- `notebooks/Preprocessing.ipynb` - exploratory preprocessing and visualization.
- `notebooks/main_with_split.ipynb` - feature extraction, grouped model
  evaluation, artifact export, and held-out-subject prediction.
- `outputs/` - saved model, predictions, comparison tables, and confusion matrix.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- The Pathological Gait dataset from
  [kooksung/pathological_gait_datasets](https://github.com/kooksung/pathological_gait_datasets)

## Setup

From the repository root, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run the commands with the environment's
interpreter directly, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Prepare the data

1. Download the dataset from
   [kooksung/pathological_gait_datasets](https://github.com/kooksung/pathological_gait_datasets).
2. Place the extracted `Pathological_Gaits` directory at:
   `data/Pathological_Gaits/`.
3. Build the balanced Parquet files from the repository root:

```powershell
.\.venv\Scripts\python.exe src\build_parquet.py
```

This creates the six gait files in `data/parquet/`, plus
`selection_manifest.csv` and `summary.csv`. The source data is not committed to
this repository.

## Run the notebooks

The notebooks use paths relative to the `notebooks` directory. Start Jupyter
from that directory:

```powershell
Set-Location notebooks
..\.venv\Scripts\jupyter.exe notebook main_with_split.ipynb
```

Alternatively, open `notebooks/main_with_split.ipynb` in VS Code and run the
cells in order with the project virtual environment selected as the kernel.
Run `Preprocessing.ipynb` first if exploratory preprocessing or plots are
needed.

The main notebook:

1. validates the Parquet schema;
2. creates one feature row per walk;
3. evaluates classifiers with subject-grouped cross-validation;
4. saves the best model and evaluation artifacts; and
5. demonstrates prediction on held-out subjects.

## Generated outputs

Running the main notebook updates or creates:

- `data/features.csv`
- `data/test_features.csv`
- `outputs/best_model.joblib`
- `outputs/model_comparison.csv`
- `outputs/model_comparison_folds.csv`
- `outputs/predictions.csv`
- `outputs/confusion_matrix.csv`
- `outputs/confusion_matrix.png`

The saved Joblib bundle contains the selected model, feature column names, and
class labels for repeatable inference.
