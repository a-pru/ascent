# ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace

### Paper
> **ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace**
> Alexander Prutsch, David Schinagl, Horst Possegger
> **Graz University of Technology**  
> **ICRA 2026**

## Getting Started

### Create and Activate Virtual Environment
```
conda create -n ascent python=3.11
conda activate ascent
```

### Install PyTorch
Our implementation was primarily developed and tested using PyTorch 2.8.0 and CUDA 12.8.  
The codebase is flexible across different configurations; for example, it has been successfully tested on PyTorch 1.11.0+cu113 and PyTorch 2.9.1+cu126.

Install PyTorch e.g.
```
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Install Dependencies
```
pip install -r ./requirements.txt
```

### TrajAir Dataset Setup
Download the Trajair dataset from [the official site](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251).  
Unzip the files in the `dataset` folder.  
The first time a dataset is loaded with a specific configuration (history length, future length, and sampling step), a cached file is saved to `dataset/_cache` to accelerate future loading times.

### Tartan Aviation Dataset Setup
** Instructions comming soon. ** 

## Training
Get available parameters: `python train.py -h`  

Run training on 7days1 split with default parameters:
`CUBLAS_WORKSPACE_CONFIG=:4096:8 && python train.py --dataset_name 7days1`    
For each run, a new directory is created in runs/ containing checkpoints, log files, model configurations, and a copy of the model implementation to ensure reproducibility.

## Evaluation
### Pretrained Models
We provide pretrained checkpoints for TrajAir split days1-4 with 16s history as input (lower group in paper Table 1) in the `checkpoints` folder.  
To run evaluations to for example:  
`python test.py --exp_folder checkpoints/model_7days1/ --epoch 11`

### Custom Runs
To evaluate a custom model, select its experiment folder and choose an epoch. You can optionally select a dataset split; if left blank, the split defaults to the one defined in the training configuration.  
`python test.py --exp_folder runs/2026-XX-XX_XX-XX-XX/ --dataset_name 7days1 --epoch 10`

## Bibtex
```bibtex
@inproceedings{prutsch2026ascent,
    title={{ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace}},
    author={Prutsch, Alexander and Schinagl, David and Possegger, Horst},
    booktitle={In Proceedings of the IEEE International Conference on Robotics and Automation},
    year={2026}
}
```

## Acknowledgements
This repository is based on [TrajAirNet](https://github.com/castacks/trajairnet). We thank them for their work!
