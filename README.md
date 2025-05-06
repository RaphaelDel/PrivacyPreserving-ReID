# FG-ReID-25
Implementation of the Paper 50 

## About

This repository contains the implementation of a novel privacy-preserving approach for person re-identification (Re-ID) using depth images. Our method leverages the inherent privacy advantages of depth data, which obscures facial and other identifiable features, offering a privacy-friendly alternative to traditional RGB-based Re-ID systems. The use of a top-down configuration further ensures sensitive personal details are not captured, balancing privacy preservation with effective behavioral data collection.

Key features of this work include the application of the Hungarian algorithm to solve the association problem by optimizing matches globally across the distance matrix, and the use of temporal sequences to capture dynamic movement patterns. These techniques significantly improve re-identification performance, particularly in challenging scenarios such as occlusions or visually similar individuals.

The approach has been validated on multiple datasets, including TVPR2, GODPR, and BIWI RGBD-ID, demonstrating competitive performance in privacy-sensitive applications like public transport systems. This repository provides the tools and code necessary for exploring robust and scalable depth-only Re-ID solutions.


## Setup

### Environment

Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

Create Conda environment:

```
conda create -n reid python=3.9

```

And activate it:

```
conda activate reid
```

Install required libraries:

```
pip install -r requirements.txt
```

### Data

In our study, we used three datasets: [TVPR2](https://vrai.dii.univpm.it/content/tvpr2-dataset), [BIWI](https://robotics.dei.unipd.it/reid/index.php/8-dataset/2-overview-biwi), and GODPR.

Each dataset requires specific preprocessing steps, which can be found in the `preprocess` folder. Please ensure that the preprocessing is completed as outlined for each dataset.

The processed data should then be placed in the `data` folder.

## Training and Evaluation

### For depth only model

The script `train_and_evaluateDepthOnly.py` needs to be executed after modifying the parameters inside it (eg. path of data, number of workers, etc).

After the training, the model will automaticly be evaluated on the test split with and without the hungarian optimisation. The results will be ploted in the `metrics` folder.

### For RGBD model

The script `train_and_evaluate.py` needs to be executed after modifying the parameters inside it (eg. path of data, number of workers, etc).

After the training, the model will automaticly be evaluated on the test split with and without the hungarian optimisation. The results will be ploted in the `metrics` folder.
