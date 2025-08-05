# PINGS-X
1. Environment Setup
    Python: 3.9.23
    PyTorch: 2.7.1+cu28

2. Install Libraries using pip

      pip install -r requirements.txt


---
PINGS-X
Training

    python train_carotid_PINGS_avg.py \
        --data_dir ./Data/carotid/MRV_2.mat \
        --output_dir ./final_results_PINGS_avg_2 \
        --epochs 10001 \
        --lr 0.01 \
        --batch_size 10000 \
        --save_step 100 \
        --dense_step 100 \
        --merge_step 100
Evaluation

    python plot_carotid_PINGS.py \
        --ckpt_path ./final_results_PINGS_avg_2_10_100_100_2/Weights/1000.tar \
        --mat_path ./Data/carotid/MRV.mat \
        --training_path ./Data/carotid/MRV_2.mat

---
PINN
Training

    python train_carotid_PINN_sun_avg.py \
        --data_dir ./Data/carotid/MRV_2.mat \
        --model_dir ./Results_carotid_PINN_sun_avg_2 \
        --lr 1e-4 \
        --epochs 100000 \
        --save_interval 10000
Evaluation

    python plot_carotid_sun_new.py \
        --model_type PINN \
        --ckpt_path ./Results_carotid_PINN_sun_avg_2/100000.tar \
        --mat_path ./Data/carotid/MRV.mat \
        --training_path ./Data/carotid/MRV_2.mat

---
Siren
Training

    python train_carotid_Siren_sun_avg.py \
        --data_dir ./Data/carotid/MRV_2.mat \
        --model_dir ./Results_carotid_Siren_sun_avg_2 \
        --lr 5e-6 \
        --epochs 100000 \
        --save_interval 10000
Evaluation

    python plot_carotid_sun_new.py \
        --model_type Siren \
        --ckpt_path ./Results_carotid_Siren_sun_avg_2/100000.tar \
        --mat_path ./Data/carotid/MRV.mat \
        --training_path ./Data/carotid/MRV_2.mat





