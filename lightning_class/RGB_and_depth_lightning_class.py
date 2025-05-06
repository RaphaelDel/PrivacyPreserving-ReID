import torch.nn as nn
import torch
from torch import optim
import torchvision.models as models
import lightning as L
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import random
import cv2
import numpy as np

from pretraitementDB import is_image_valid, pre_processing

MAX_SEQUENCE_LENGHT = 1

class DepthVRAI(Dataset):
    def __init__(self, img_labels, img_dir, train=True, transformDepth=None, transformRGB=None,
                normaliseDepth = True, substract_Depth=True, maxSequenceLenght = MAX_SEQUENCE_LENGHT):
        self.is_train = train
        self.transformDepth = transformDepth
        self.transformRGB = transformRGB
        self.normaliseDepth = normaliseDepth
        self.substract_Depth = substract_Depth

        self.img_labels = img_labels
        self.index = self.img_labels.index.values
        self.img_dir = img_dir
        self.maxSequenceLenght = maxSequenceLenght

    def __len__(self):
        return len(self.img_labels)
    
    def substract_depth(self, depthimg, RGBimg, maxdepth=0.2, isTensor=False):
        # Créer un masque où les valeurs de profondeur sont supérieures à maxdepth
        mask = (depthimg > maxdepth) | (depthimg ==0)

        # Étendre le masque pour couvrir les 3 canaux RGB
        if isTensor:
            mask_rgb = np.expand_dims(np.repeat(mask[np.newaxis, :, :], 3, axis=0), axis=0)
        else:
            mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Appliquer le masque à l'image RGB et mettre à zéro les valeurs correspondantes
        RGBimg[mask_rgb] = 0
        return RGBimg

    def sequence_imgs(self, abs_path, passage_path,  direction):

        images = os.listdir(os.path.join(abs_path, passage_path))
        images_depth = [im for im in images if "_depth_depth" in im]

        d_list_frames = []
        rgb_list_frames = []
        for nom_image in images_depth:
            d_img = cv2.imread(os.path.join(abs_path, passage_path, nom_image), cv2.IMREAD_UNCHANGED)
            if is_image_valid(d_img):
                d_list_frames.append(d_img)
                rgb_img = cv2.imread(os.path.join(abs_path, passage_path, nom_image.replace("_depth_depth", "_RGB_person")), cv2.IMREAD_UNCHANGED)
                rgb_list_frames.append(rgb_img)

        sum_before = 0
        freq = len(d_list_frames) // (self.maxSequenceLenght + 1)
        list_position_depth = []
        for i in range(int(self.maxSequenceLenght)):
            sum_before += freq
            list_position_depth.append(round(sum_before))

        assert len(list_position_depth) == self.maxSequenceLenght

        depth_sequence = []
        rgb_sequence = []
        for i in list_position_depth:
            depthTmpImg = pre_processing(d_list_frames[i], direction, normalise=self.normaliseDepth)
            depth_sequence.append(depthTmpImg)
            if self.substract_Depth == True:
                rgb_sequence.append(self.substract_depth(depthTmpImg, pre_processing(rgb_list_frames[i], direction, normalise=False), isTensor=False))
            else:
                rgb_sequence.append(pre_processing(rgb_list_frames[i], direction, normalise=False))

        if self.maxSequenceLenght == 1:
            depth_sequence = depth_sequence[0]
            rgb_sequence = rgb_sequence[0]

        return depth_sequence, rgb_sequence

    def __getitem__(self, idx):
        anchor_path = self.img_labels.iloc[idx, 1]
        anchor_depth_sequence, anchor_rgb_sequence = self.sequence_imgs(self.img_dir, anchor_path, 0)

        # le positif correspond au OUT de l idx, soit le champ 1
        positive_path = self.img_labels.iloc[idx, 2]
        postive_depth_sequence, postive_rgb_sequence = self.sequence_imgs(self.img_dir, positive_path, 1)

        if self.is_train:
            # Peut etre un reshape ????

            # le negatif correspond au OUT d un autre idx, champ 1 egalement
            negative_list = self.index[self.index!=idx]
            negative_item = random.choice(negative_list)
            negative_path = self.img_labels.iloc[negative_item, 2]
            negative_depth_sequence, negative_rgb_sequence = self.sequence_imgs(self.img_dir, negative_path, 1)

            if self.transformDepth:
                if type(anchor_depth_sequence)  == list:
                    anchor_depth_sequence = torch.stack([self.transformDepth(img) for img in anchor_depth_sequence], 0)
                    postive_depth_sequence = torch.stack([self.transformDepth(img) for img in postive_depth_sequence], 0)
                    negative_depth_sequence = torch.stack([self.transformDepth(img) for img in negative_depth_sequence], 0)

                    anchor_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in anchor_rgb_sequence], 0)
                    postive_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in postive_rgb_sequence], 0)
                    negative_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in negative_rgb_sequence], 0)
                else:
                    anchor_depth_sequence = self.transformDepth(anchor_depth_sequence)
                    postive_depth_sequence = self.transformDepth(postive_depth_sequence)
                    negative_depth_sequence = self.transformDepth(negative_depth_sequence)

                    anchor_rgb_sequence = self.transformRGB(anchor_rgb_sequence)
                    postive_rgb_sequence = self.transformRGB(postive_rgb_sequence)
                    negative_rgb_sequence = self.transformRGB(negative_rgb_sequence)
            # return les trois imgs et le label de l ancre
            return (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence), (negative_depth_sequence, negative_rgb_sequence), idx

        else:
            if self.transformDepth:
                if type(anchor_depth_sequence)  == list:
                    anchor_depth_sequence = torch.stack([self.transformDepth(img) for img in anchor_depth_sequence], 0)
                    postive_depth_sequence = torch.stack([self.transformDepth(img) for img in postive_depth_sequence], 0)

                    anchor_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in anchor_rgb_sequence], 0)
                    postive_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in postive_rgb_sequence], 0)
                else:
                    anchor_depth_sequence = self.transformDepth(anchor_depth_sequence)
                    postive_depth_sequence = self.transformDepth(postive_depth_sequence)

                    anchor_rgb_sequence = self.transformRGB(anchor_rgb_sequence)
                    postive_rgb_sequence = self.transformRGB(postive_rgb_sequence)
            return (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence)


class TripletLoss(nn.Module):
    def __init__(self, margin=256.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


# --------------------------------------------------------------------
# SEQUENCE ENCODER

class SequenceDepthEncoder(nn.Module):
    def __init__(self, embedding_size, NbFrames):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(1, 64, 3),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2,stride=2))
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 512, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.encoder = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*3*3, out_features=embedding_size))
    
    def forward(self, image_sequence):
        # From Batch, Seq, Filter, H, W to  : Seq, Batch, Filter, H, W 
        embeddings = []
        for img in image_sequence:
            x = self.conv_layer_1(img)
            x = self.conv_layer_2(x)
            x = self.conv_layer_3(x)
            x = self.conv_layer_3(x)
            x = self.conv_layer_3(x)
            x = self.conv_layer_3(x)
            x = self.encoder(x)
            embeddings.append(x)
        embeddings = torch.stack(embeddings)  # Convert list of tensors to a tensor
        # OutPut size : Seq, Batch, EmbeddSize

        # Mean
        transformer_output = embeddings.mean(0)
        return transformer_output

class SequenceRGBEncoder(nn.Module):
    def __init__(self, embedding_size, NbFrames):
        super().__init__()

        # Charger ResNet50 pré-entraîné
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        last_hiden_layer_size = self.resnet50.fc.in_features

        # Supprimer la dernière couche (la couche de classification)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Ajouter une nouvelle couche linéaire pour obtenir l'embedding
        self.embedding_layer = nn.Sequential(
            nn.Linear(last_hiden_layer_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size),
            nn.ReLU()
        )
    
    def freeze_feature_extractor(self, layers_not_freeze = 4):
        # Gel de tous les paramètres du feature_extractor sauf les n derniers layers
        layers = list(self.resnet50.children())  # Obtenir tous les layers du feature_extractor
        layers_to_freeze = layers[:-layers_not_freeze]  # Tous sauf les n derniers layers

        # Geler les paramètres des layers sélectionnés
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):

        embeddings = []
        for img in x:
            # Passer l'image à travers ResNet50 (partie feature extraction)
            features = self.resnet50(img)
            # Aplatir les caractéristiques
            features = features.view(features.size(0), -1)
            # Obtenir l'embedding
            embedding = self.embedding_layer(features)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)  # Convert list of tensors to a tensor
        # OutPut size : Seq, Batch, EmbeddSize

        # Mean
        output = embeddings.mean(0)
        return output

class SequenceEncoder(L.LightningModule):
    def __init__(self, embedding_size, NbFrames, Nb_layerToFreeze=8):
        super().__init__()

        # Depth Sequence Encoder
        self.DepthSeqEncoder = SequenceDepthEncoder(embedding_size, NbFrames)
        self.RGBSeqEncoder = SequenceRGBEncoder(embedding_size, NbFrames)
        self.RGBSeqEncoder.freeze_feature_extractor(Nb_layerToFreeze)
        self.loss = TripletLoss(margin=embedding_size)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, depth_sequence, rgb_sequence):
        # From Batch, Seq, Filter, H, W to  : Seq, Batch, Filter, H, W 
        depth_sequenceP = depth_sequence.permute(1, 0, 2, 3, 4)
        # print("depth_sequenceP.size()", depth_sequenceP.size())
        rgb_sequenceP = rgb_sequence.permute(1, 0, 2, 3, 4)
        # print("rgb_sequenceP.size()", rgb_sequenceP.size())

        depth_encoded = self.DepthSeqEncoder(depth_sequenceP)
        rgb_encoded = self.RGBSeqEncoder(rgb_sequenceP)
        # print("depth_encoded.size", depth_encoded.size())
        # print("rgb_encoded.size", rgb_encoded.size())
        
        #Use to have RGB and Depth information at the same time
        #output = torch.cat((depth_encoded, rgb_encoded), dim=-1)
        
        #rgb only
        output = depth_encoded
        # print("output.size", output.size())
        return output

    def training_step(self, batch, batch_idx):

        (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence), (negative_depth_sequence, negative_rgb_sequence), idx = batch
        # Obtenir l'embedding à partir de l'image

        anchor_out = self.forward(anchor_depth_sequence, anchor_rgb_sequence)
        positive_out = self.forward(postive_depth_sequence, postive_rgb_sequence)
        negative_out = self.forward(negative_depth_sequence, negative_rgb_sequence)

        train_loss = self.loss(anchor_out, positive_out, negative_out)

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)

        correct = (d_ap < d_an).sum()
        total = len(d_an)

        self.log("Train/train_loss", train_loss, prog_bar=True)
        self.training_step_outputs.append({"loss" : train_loss, "correct" : correct, "total" : total})
        return train_loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()

        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in self.training_step_outputs])
        total=sum([x["total"] for  x in self.training_step_outputs])

        self.log_dict({"Train/avg_loss" : avg_loss, "Train/Accuracy" : correct/total}, prog_bar=True)

        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):

        (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence), (negative_depth_sequence, negative_rgb_sequence), idx = batch
        # Obtenir l'embedding à partir de l'image

        anchor_out = self.forward(anchor_depth_sequence, anchor_rgb_sequence)
        positive_out = self.forward(postive_depth_sequence, postive_rgb_sequence)
        negative_out = self.forward(negative_depth_sequence, negative_rgb_sequence)
        val_loss = self.loss(anchor_out, positive_out, negative_out)

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)

        correct = (d_ap < d_an).sum()
        total = len(d_an)

        if batch_idx == 1: # Log every 10 batches
            self.log_tb_images([anchor_depth_sequence, postive_depth_sequence, negative_depth_sequence, d_ap, d_an])

        self.log("Val/val_loss", val_loss, prog_bar=True)

        self.validation_step_outputs.append({"loss" : val_loss, "correct" : correct, "total" : total})

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()

        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in self.validation_step_outputs])
        total=sum([x["total"] for  x in self.validation_step_outputs])

        self.log_dict({"Val/avg_loss" : avg_loss, "Val/Accuracy" : correct/total}, prog_bar=True)

        self.validation_step_outputs.clear()  # free memory

    def log_tb_images(self, viz_batch):

        # Get tensorboard logger
        tb_logger = self.logger.experiment

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')

        # Log the images (Give them different names)
        for img_idx, a in enumerate(zip(*viz_batch)):
            anchor_img, positive_img, negative_img, d_ap, d_an = a
            batch_idx = 0

            anchorToPrint = torch.cat(torch.unbind(anchor_img, 0), -1)
            positiveToPrint = torch.cat(torch.unbind(positive_img, 0), -1)
            negativeToPrint = torch.cat(torch.unbind(negative_img, 0), -1)
            images_concat = torch.cat([anchorToPrint, positiveToPrint, negativeToPrint], -2)

            tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", images_concat, 0)
            # tb_logger.add_image(f"ImagePositive/{batch_idx}_{img_idx}", positive_img, 0)
            # tb_logger.add_image(f"PositiveDistance/{batch_idx}_{img_idx}", d_ap, 0)
            # tb_logger.add_image(f"ImageNegative/{batch_idx}_{img_idx}", negative_img, 0)
            # tb_logger.add_image(f"NegativeDistance/{batch_idx}_{img_idx}", d_an, 0)

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer