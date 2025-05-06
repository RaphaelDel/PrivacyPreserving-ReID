import sys
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import lightning as L

from sklearn.model_selection import train_test_split
from lightning_class import batch_hard_Transformer as Lc
from lightning_class import RGB_and_depth_lightning_class as rgb_Lc
from lightning.pytorch.callbacks import ModelCheckpoint

# Dataset
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger

# Evaluation
from tqdm import tqdm
import eval_metrics

# Random seed
torch.manual_seed(42)
np.random.seed(42)

# CONSTANTES
NBFRAMES = 4
IMGVARIATIONS = 2
EMBEDDING_SIZE = 128
BATCH_SIZE = 5
NUM_WORKER = 10
MAX_EPOCHS = 10

#RGB SeqEncoder
Nb_layerNotToFreeze=4

def trainTestSplit(annotations_file):

    img_labels = pd.read_csv(annotations_file)

    train, val = train_test_split(img_labels, test_size=0.2, random_state=42)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    return train, val

# Pre traitement des images en entree
RGBTransform = transforms.Compose([
    transforms.ToTensor(), # Convertir l'image en tenseur PyTorch
    transforms.Resize(256),  # Redimensionner l'image pour qu'elle soit suffisamment grande
    transforms.CenterCrop(224),  # Rogner l'image au centre à la dimension requise par ResNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normaliser l'image
])

depthTransform = transforms.Compose([
    transforms.ToTensor(), # Convertir l'image en tenseur PyTorch
    transforms.Resize(256),  # Redimensionner l'image pour qu'elle soit suffisamment grande
    transforms.CenterCrop(224),  # Rogner l'image au centre à la dimension requise par ResNet
    transforms.Normalize(mean=0, std=0.25),  # Normaliser l'image
])

def trainModel(train, val, log_folder):

    train_datasetD = Lc.BatchHardVRAI(img_labels=train, img_dir=r"../../data/nimages",
                                    train=True, transformDepth=depthTransform, normaliseDepth = True,
                                    transformRGB=RGBTransform, substract_Depth=True,
                                    maxSequenceLenght=NBFRAMES, nb_images_variations=IMGVARIATIONS)
    
    
    val_datasetD = Lc.BatchHardVRAI(img_labels=val, img_dir=r"../../data/nimages",
                                    train=True, transformDepth=depthTransform, normaliseDepth = True,
                                    transformRGB=RGBTransform, substract_Depth=True,
                                    maxSequenceLenght=NBFRAMES, nb_images_variations=IMGVARIATIONS)
    
    train_dataloaderD = DataLoader(train_datasetD, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKER, persistent_workers=True)
    val_dataloaderD = DataLoader(val_datasetD, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKER, persistent_workers=True)

    seq_encoder = Lc.SequenceEncoder(EMBEDDING_SIZE, NbFrames=NBFRAMES, Nb_layerToFreeze=Nb_layerNotToFreeze)

    logger = TensorBoardLogger(log_folder, name='RGBD_Transformer_HB') #, version='MeanOverSeq'
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="Val/val_loss")

    trainerD = L.Trainer(accelerator="gpu", max_epochs=MAX_EPOCHS, logger=logger, profiler="simple", callbacks=[checkpoint_callback]) # , profiler="simple" for inspecting bottleneck
    trainerD.fit(model=seq_encoder, train_dataloaders=train_dataloaderD, val_dataloaders=val_dataloaderD)

    return seq_encoder

    #### ------------------------------------------------------------------------------------------------
    # Partie Evaluation

def evaluate(depth_encoder, folder_to_save='./metrics', device='cuda:0'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_encoder.to(device)
    depth_encoder.eval()

    data_val = rgb_Lc.DepthVRAI(img_labels=val, img_dir=r"../../data/nimages", train=False, transformDepth=depthTransform, normaliseDepth = True, transformRGB=RGBTransform, maxSequenceLenght=NBFRAMES)

    tab_resnet = []
    for i in tqdm(range(len(data_val))):
        with torch.no_grad():
            (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence) = data_val[i]
            anchor, _, _ = depth_encoder(torch.unsqueeze(anchor_depth_sequence, 0).to(device), torch.unsqueeze(anchor_rgb_sequence, 0).to(device))
            positive, _, _ = depth_encoder(torch.unsqueeze(postive_depth_sequence, 0).to(device), torch.unsqueeze(postive_rgb_sequence, 0).to(device))

            tab_resnet.append([anchor, positive])

    eval_metrics.plot_and_save(tab_resnet, folder_to_save)

if __name__ == '__main__':

    annotations_file = r"../../labels_csv.csv"
    train, val = trainTestSplit(annotations_file)

    log_folder = r"./tb_logs"
    model = trainModel(train, val, log_folder)

    evaluate(model)