# LSGNN course project

Reimplementation of **LSGNN** for experiments.

## Layout

- `src/data_processing/` — preprocess, load, splits, features, I/O
- `src/models/` — `model.py`, `train.py`
- `src/evaluation/` — `evaluate.py` (library + CLI), `run_evaluate.py` (lightweight checkpoint eval)
- `src/utils/` — `helpers.py`, `log_setup.py`, `cal_mean_metric.py`
- `data/raw/`, `data/processed/`
- `results/model_outputs/<dataset>/` — checkpoints (`.ckpt`)
- `results/plots/<dataset>/` — training curve PNGs
- `results/logs/<dataset>/` — `train.log`, `preprocess.log`, etc.
- `notebooks/` — `model_training.ipynb`, `data_analysis.ipynb`
- `tests/` — small smoke scripts (`test_model.py`, `test_preprocessing.py`)

### One-shot root scripts (preprocess, then train + final metrics)

Create and activate your own conda/venv and install PyTorch + PyG + deps per [Setup](#setup). Then:

**Highly recommand that create the environment and install all dependencies, then run these scripts**

1. **Preprocess all datasets** (optional `pip install -r requirements.txt` if you pass `--install`):

   ```bash
   ./setup_and_preprocess.sh --install
   ```

   Dependencies already installed:

   ```bash
   ./setup_and_preprocess.sh
   ```

2. **Train everything and write a summary file** (long run). Split counts per dataset are listed in `run_train_and_results.sh` and must match your `data/processed/*.pt`:

   ```bash
   ./run_train_and_results.sh
   ```

   Aggregated console output is also saved to `results/FINAL_RESULTS.txt`.

## Setup

Use **Python 3.11** (3.10+ is usually fine).

**Important:** Install **PyTorch** first, then **PyG extension wheels** that match your exact `torch` build (`torch-2.Y.Z+cuXXX` or `+cpu`), then `torch-geometric`, then `pip install -r requirements.txt`.  
See the [PyG installation page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and pick the wheel index URL that matches `python -c "import torch; print(torch.__version__)"`.

The commands below use **CUDA 12.8 (`cu128`)** with **PyTorch 2.11.x** and matching PyG wheels(This setup works on my personal PC with a RTX 5070). Replace `torch-2.11.0+cu128.html` (or `+cpu.html`) with the [PyG wheel page](https://data.pyg.org/whl/) that matches `python -c "import torch; print(torch.__version__)"` if yours differs (e.g. `cu124` / `cu121`/ `cu118`).

**Note:** If `torch_spline_conv` has no prebuilt wheel for your torch version, skip it — this project does not require it for training Planetoid / Geom-GCN style graphs.

Without a GPU, training and evaluation still run on **CPU** (slower).

### Using Conda

**CUDA 12.8 (`cu128`)** and **PyTorch 2.11.x** (tested):

```bash
conda create -n lsgnn python=3.11 -y
conda activate lsgnn

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

pip install torch-geometric
pip install -r requirements.txt
```

**CPU-only** (adjust the PyG URL to `torch-X.Y.Z+cpu.html` matching your `torch`):

```bash
conda create -n lsgnn python=3.11 -y
conda activate lsgnn

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cpu.html

pip install torch-geometric
pip install -r requirements.txt
```

### Using venv (no Conda)

You need a **Python 3.11+** binary on your PATH (Path variable) before run these commands. This setup is for you if you can;t or dont have conda

```bash
python3.11 -m venv lsgnn-venv
source lsgnn-venv/bin/activate
pip install -U pip
```
**source** command is for linux, if you run on a different OS, please search for how to create a venv on that OS(I have on clue how to do these on Windows and MacOS)

Then install the same packages as in Conda (**CUDA 12.8** example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

pip install torch-geometric
pip install -r requirements.txt
```

Or **CPU-only**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cpu.html

pip install torch-geometric
pip install -r requirements.txt
```

Point **Jupyter / VS Code / Cursor** at `lsgnn-venv/bin/python` (or `lsgnn-venv\Scripts\python.exe`) and install `ipykernel` if notebooks should use this env (`pip install ipykernel` is already listed in `requirements.txt`).

## Run 

```bash
cd /path/to/Guanhua_Zhao
```


### Preprocess

```bash
python -m src.data_processing.preprocess --dataset cora
```
The preprocess might get stuck at the end of preprocess, please terminate it manully after it saves the preprocessed data


### Train

`--split-idx` selects which column of the multi-split masks in `data/processed/<name>.pt` to use (default **0**). Checkpoint and plot filenames include `_split<K>`. 

```bash
python -m src.models.train --dataset cora --processed-name cora --split-idx 0
```

If you want to train all datasets with all splits, code will be given at the end of this file.


### Evaluate (full logging to `results/logs/<dataset>/evaluate.log`)

Omit `--split-idx` to use the split stored in the checkpoint config; pass it to override (must match the masks used for that run).

```bash
python -m src.evaluation.evaluate \
  --ckpt results/model_outputs/cora/lsgnn_cora_split0_best_loss.ckpt \
  --processed-name cora \
  --split-idx 0
```

### Evaluate (lightweight; prints metrics only)

```bash
python -m src.evaluation.run_evaluate \
  --ckpt results/model_outputs/cora/lsgnn_cora_split0_best_loss.ckpt \
  --processed-name cora \
  --split-idx 0
```

### Batch preprocess

```bash
bash preprocess_all.sh
```

### Smoke checks (optional)

```bash
python tests/test_model.py
python tests/test_preprocessing.py
```

### Aggregate metrics from logs

```bash
python -m src.utils.cal_mean_metric --dataset cora --last-n 10
```

## Where to edit

- Model: `src/models/model.py`
- Training config: `default_lsgnn_config` in `src/models/train.py`, plus `src/configs/<dataset>.json`
- Data / splits: `src/data_processing/splits.py`, `load_data.py`, `preprocess.py`



## My personal bash commands used: preprocess + train + metrics

Run from **`Guanhua_Zhao` repo root**. Skip any `preprocess` line if `data/processed/<name>.pt` already exists. For ogbn-arxiv, add `--raw-root ...` to `preprocess` if OGB data is not under `data/raw`.

```bash
set -e
# cora 
python -m src.data_processing.preprocess --dataset cora
for i in $(seq 0 9); do python -m src.models.train --dataset cora --processed-name cora --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset cora

#  citeseer 
python -m src.data_processing.preprocess --dataset citeseer
for i in $(seq 0 9); do python -m src.models.train --dataset citeseer --processed-name citeseer --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset citeseer

#  pubmed 
python -m src.data_processing.preprocess --dataset pubmed
for i in $(seq 0 9); do python -m src.models.train --dataset pubmed --processed-name pubmed --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset pubmed

#  chameleon 
python -m src.data_processing.preprocess --dataset chameleon
for i in $(seq 0 9); do python -m src.models.train --dataset chameleon --processed-name chameleon --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset chameleon

#  squirrel 
python -m src.data_processing.preprocess --dataset squirrel
for i in $(seq 0 9); do python -m src.models.train --dataset squirrel --processed-name squirrel --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset squirrel

#  actor 
python -m src.data_processing.preprocess --dataset actor
for i in $(seq 0 9); do python -m src.models.train --dataset actor --processed-name actor --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset actor

#  cornell 
python -m src.data_processing.preprocess --dataset cornell
for i in $(seq 0 9); do python -m src.models.train --dataset cornell --processed-name cornell --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset cornell

#  texas 
python -m src.data_processing.preprocess --dataset texas
for i in $(seq 0 9); do python -m src.models.train --dataset texas --processed-name texas --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset texas

#  wisconsin 
python -m src.data_processing.preprocess --dataset wisconsin
for i in $(seq 0 9); do python -m src.models.train --dataset wisconsin --processed-name wisconsin --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset wisconsin

#  ogbn-arxiv:Large graph one split
python -m src.data_processing.preprocess --dataset ogbn-arxiv
python -m src.models.train --dataset ogbn-arxiv --processed-name ogbn-arxiv --split-idx 0
python -m src.utils.cal_mean_metric --dataset ogbn-arxiv --last-n 1

#  arxiv-year:Large graph 5 splits
python -m src.data_processing.preprocess --dataset arxiv-year
for i in $(seq 0 4); do python -m src.models.train --dataset arxiv-year --processed-name arxiv-year --split-idx $i; done
python -m src.utils.cal_mean_metric --dataset arxiv-year --last-n 5
```

Remove `set -e` if you prefer one failure not to stop the whole script. Copy only the blocks you need.

## Results : Including some test on paper original code

### Table 1 (nine small graphs, 10 splits each)

| Dataset | Paper LSGNN (%) |LSGNN v2 run by me (test mean ± std)| This project (accuracy) | This project run time (s, mean ± std) |
|--------|-----------------|-----------------------------------------------|-----------------------------------------------------|-------------------------------------|
| Cora | 88.49 | **88.14 ± 1.27** | **87.99 ± 1.44** (val 89.58 ± 0.63) | **4.27 ± 2.17** |
| Citeseer | 76.71 | **77.43 ± 1.16** | **76.66 ± 1.29** (val 76.75 ± 0.81) | **4.20 ± 1.28** |
| Pubmed | 90.23 | **89.81 ± 0.31** | **89.89 ± 0.55** (val 90.06 ± 0.23) | **3.92 ± 0.44** |
| Chameleon | 79.04 | **78.82 ± 0.85** | **78.68 ± 1.10** (val 77.79 ± 1.15) | **5.33 ± 7.21** |
| Squirrel | 72.81 | **72.79 ± 2.09** | **72.57 ± 2.12** (val 73.17 ± 0.76) | **6.91 ± 4.85** |
| Actor | 36.18 | **35.16 ± 1.20** | **35.10 ± 1.08** (val 36.97 ± 0.96) | **2.87 ± 0.12** |
| old Cornell | 88.92 | **76.22 ± 4.65** | **82.70 ± 5.95** (val 90.17 ± 3.70) | **1.46 ± 0.14** |
| Texas | 90.27 | **90.00 ± 2.97** | **91.08 ± 4.53** (val 91.02 ± 4.68) | **1.42 ± 0.10** |
| Wisconsin | 90.20 | **86.47 ± 4.34** | **86.47 ± 2.70** (val 89.00 ± 2.84) | **1.47 ± 0.15** |
| **Avg** (paper column) | 79.21 | **78.31** (mean of 9 official test means) | — | **3.54** |

### Table 2 (large graphs)


| Dataset | LSGNN v2 run by me (test mean ± std) |This project (accuracy) | This project run time (s, mean ± std) |
|--------|--------------------------|-----------|-------------------------------------|
| ogbn-arxiv | 72.64 | **72.11** (val 72.88; n=1) | **77.55 ± 0.00** (n=1) |
| arxiv-year | 56.42 | **50.90 ± 0.30** (val 50.99 ± 0.15) | **42.95 ± 1.79** |
| **Avg** (paper row) | 64.53 | — | — |