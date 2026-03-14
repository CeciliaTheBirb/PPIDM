# PPIDM

**Partial Physics-Informed Diffusion Model for Ocean Chlorophyll Concentration Reconstruction**

This repository contains the official implementation of **PPIDM** (NIPS 2025), a partial physics informed diffusion model for long range imputation and prediction of ocean chlorophyll concentration fields.


## Data Preparation

The model is trained and evaluated using data from the Biogeochemical Southern Ocean State Estimate (B-SOSE). You can access the data [here](http://sose.ucsd.edu/BSOSE_iter105_solution.html).

To train the model, please download the following specific datasets:
* **`Zonal Component of Velocity`** (3 day average)
* **`Meridional Component of Velocity`** (3 day average)
* **`Chlorophyll concentration`** (3 day average)

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/CeciliaTheBirb/PPIDM.git
cd ppidm
pip install -r requirements.txt
```

## Training

The training and sampling procedures are built upon the [RaMViD](https://github.com/Tobi-r9/RaMViD) framework. To train the PPIDM model, we follow the baseline flags:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 4 --microbatch 2 --seq_len 20 --max_num_mask_frames 4 --uncondition_rate 0.5";
```

Then, initiate the training script by pointing to your prepared B-SOSE data directory:

```bash
python scripts/nc_train.py --chl_dir /PATH/TO/bsose_i105_2008to2012_3day_Chl.nc --u_dir /PATH/TO/bsose_i105_2008to2012_3day_Uvel.nc --v_dir /PATH/TO/bsose_i105_2008to2012_3day_Vvel.nc $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```


## Sampling

To generate chlorophyll reconstructions from a trained model checkpoint, use the sampling script. It is highly recommended to sample from the EMA (Exponential Moving Average) models (e.g., `ema_0.9999_200000.pt`), as they produce significantly better results.

```bash
python scripts/nc_sample.py --model_paths /PATH/TO/ema.pt --chl_dir /PATH/TO/bsose_i105_2008to2012_3day_Chl.nc --u_dir /PATH/TO/bsose_i105_2008to2012_3day_Uvel.nc --v_dir /PATH/TO/bsose_i105_2008to2012_3day_Vvel.nc --cond_frames 0,19 $MODEL_FLAGS $DIFFUSION_FLAGS
```

The generated samples will be saved in the logging directory as a large `.npz` file. 

## Citation

If you find this code or our methodology helpful for your research, please consider citing our work:

```bibtex
@inproceedings{xu2025ppidm,
  title={Partial Physics-Informed Diffusion Model for Ocean Chlorophyll Concentration Reconstruction},
  author={Xu, Qianxun and Li, Zuchuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
