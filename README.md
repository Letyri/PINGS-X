# PINGS-X: 4D Carotid Flow Modeling with PINNs & SIREN

> **PINGS-X** provides a reproducible end‑to‑end pipeline for learning and evaluating 4D carotid flow with **PINNs** and **SIREN**. The repository includes unified CLI arguments, consistent checkpointing, and example scripts for training and evaluation.

## ✨ Key Features
- **Three training pipelines**: PINGS-X, PINN, and SIREN.
- **Unified interface**: Consistent dataset paths and checkpoint structure across methods.
- **Scripted reproducibility**: Clear step‑by‑step commands from install → train → evaluate.

---

## 🖥️ Environment
- **Python**: 3.9.23
- **PyTorch**: 2.7.1+cu128

> *Adjust Python/PyTorch versions as needed for your CUDA/driver. The versions above reflect the reference setup used in this repository.*

### Installation
```bash
pip install -r requirements.txt

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 🗂️ Project Structure
- `4D_carotid/` — Training & evaluation scripts for PINGS-X, PINN, SIREN.
- `Data/carotid/` — Example input `.mat` files (`MRV.mat`, `MRV_2.mat`).

> **Note:** You can override dataset and output locations via the CLI arguments shown below.

---

## 🚀 Quick Start

### 0) Common working directory
```bash
cd ./4D_carotid
```

---

### 1) PINGS-X

**Training**
```bash
python train_carotid_PINGS_avg.py \
  --data_dir ../Data/carotid/MRV_2.mat \
  --output_dir ./final_results_PINGS_avg_2 \
  --epochs 1000 \
  --lr 0.01 \
  --batch_size 10000 \
  --save_step 100 \
  --dense_step 100 \
  --merge_step 100
```

**Evaluation**
```bash
python plot_carotid_PINGS.py \
  --ckpt_path ./final_results_PINGS_avg_2_10_100_100_2/Weights/1000.tar \
  --mat_path ../Data/carotid/MRV.mat \
  --training_path ../Data/carotid/MRV_2.mat
```

---

### 2) PINN

**Training**
```bash
python train_carotid_PINN_avg.py \
  --data_dir ../Data/carotid/MRV_2.mat \
  --model_dir ./Results_carotid_PINN_avg_2 \
  --lr 1e-4 \
  --epochs 100000
```

**Evaluation**
```bash
python plot_carotid_PINN_Siren.py \
  --model_type PINN \
  --ckpt_path ./Results_carotid_PINN_avg_2/100000.tar \
  --mat_path ../Data/carotid/MRV.mat \
  --training_path ../Data/carotid/MRV_2.mat
```

---

### 3) SIREN

**Training**
```bash
python train_carotid_Siren_avg.py \
  --data_dir ../Data/carotid/MRV_2.mat \
  --model_dir ./Results_carotid_Siren_avg_2 \
  --lr 5e-6 \
  --epochs 100000
```

**Evaluation**
```bash
python plot_carotid_PINN_Siren.py \
  --model_type Siren \
  --ckpt_path ./Results_carotid_Siren_avg_2/100000.tar \
  --mat_path ../Data/carotid/MRV.mat \
  --training_path ../Data/carotid/MRV_2.mat
```

---

## 🧪 Repro Tips
- **GPU memory**: On a single GPU, reduce `--batch_size` if you hit OOM (e.g., from `10000` down to `2000`).
- **Path consistency**: Keep `--data_dir`, `--mat_path`, and `--training_path` aligned to your actual dataset locations.
- **Logging & checkpoints**: Tune `--save_step`, `--dense_step`, `--merge_step` to balance logging granularity and storage (PINGS-X).

---

## 📈 Results (placeholder)
Add your quantitative metrics (e.g., PSNR/SSIM, physical constraint violation rate) and visualizations (flow/velocity fields) here.

---

## 📎 Citation (placeholder)
If you publish with this code, add BibTeX or paper references here.

---

## 📄 License (placeholder)
Specify the license (e.g., MIT, Apache-2.0).
