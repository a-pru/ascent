# ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace

### Paper
> **ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace**
> Alexander Prutsch, David Schinagl, Horst Possegger
> **Graz University of Technology**  
> **ICRA 2026**

# Dataset Setup
## TrajAir
Download the Trajair dataset from [the official site](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251).  
Unzip the files in the `dataset` folder.  
The first time a dataset is loaded with a specific configuration (history length, future length, and sampling step), a cached file is saved to `dataset/_cache` to accelerate future loading times.

## Tartan Aviation
** Comming soon ** 

# Training
Get available parameters: `python train.py -h`  

Run training on 7days1 split with default parameters:
`CUBLAS_WORKSPACE_CONFIG=:4096:8 && python train.py --dataset_name 7days1`    
For each run, a new directory is created in runs/ containing checkpoints, log files, model configurations, and a copy of the model implementation to ensure reproducibility.

# Evaluation
To evaluate a model, select its experiment folder and choose an epoch. You can optionally select a dataset split; if left blank, the split defaults to the one defined in the training configuration.  
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
