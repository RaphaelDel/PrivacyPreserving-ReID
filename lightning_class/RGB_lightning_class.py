from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
import lightning as L
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import numpy as np
import pandas as pd
import random
import math
import cv2

from REID_pipeline import select_frame
from preprocessRGB import substract_depth
import pretraitementDB


class VRAI(Dataset):
    def __init__(self, img_labels, img_dir, train=True, transform=None, substractDepth = False):
        self.is_train = train
        self.transform = transform

        self.img_labels = img_labels
        self.index = self.img_labels.index.values
        self.img_dir = img_dir
        self.substract_depth = substractDepth

    def __len__(self):
        return len(self.img_labels)

    def bgr_to_rgb(self, bgr_img):
        permute = [2, 1, 0]

        rgb_img = bgr_img[permute]
        return rgb_img

    def select_frame_bypath(self, abs_path, passage_path):

        images = os.listdir(os.path.join(abs_path, passage_path))
        images_depth = [im for im in images if "_D" in im]
        # max_frame = select_frame(images_depth, os.path.join(abs_path, passage_path))
        max_frame = images_depth[len(images_depth)//2]
        max_frame_RGB = max_frame.replace("_D.", "_RGB.")
        # print(max_frame_RGB)
        return max_frame_RGB, max_frame

    def __getitem__(self, idx):
        anchor_path = self.img_labels.iloc[idx, 1]
        anchor_img_path_RGB, anchor_img_path_Depth  = self.select_frame_bypath(self.img_dir, anchor_path)
        # Utilisation de cv2 pour charger les images, bon format direct et permet les transforms plus simplement que read image de torch
        anchor_img = cv2.imread(os.path.join(self.img_dir, anchor_path, anchor_img_path_RGB), cv2.IMREAD_UNCHANGED)
        # anchor_img = read_image(os.path.join(self.img_dir, anchor_path, anchor_img_path_RGB))
        # anchor_img = self.bgr_to_rgb(anchor_img)
        if self.substract_depth:
            anchor_img = substract_depth(cv2.imread(os.path.join(self.img_dir, anchor_path, anchor_img_path_Depth), cv2.IMREAD_UNCHANGED), anchor_img, isTensor=False)

        # le positif correspond au OUT de l idx, soit le champ 1
        positive_path = self.img_labels.iloc[idx, 2]
        positive_img_path_RGB, positive_img_path_Depth = self.select_frame_bypath(self.img_dir, positive_path)
        positive_img = cv2.imread(os.path.join(self.img_dir, positive_path, positive_img_path_RGB), cv2.IMREAD_UNCHANGED)
        if self.substract_depth:
            positive_img = substract_depth(cv2.imread(os.path.join(self.img_dir, positive_path, positive_img_path_Depth), cv2.IMREAD_UNCHANGED), positive_img, isTensor=False)

        if self.is_train:
            # Peut etre un reshape ????

            # le negatif correspond au OUT d un autre idx, champ 1 egalement
            negative_list = self.index[self.index!=idx]
            negative_item = random.choice(negative_list)
            negative_path = self.img_labels.iloc[negative_item, 2]
            negative_img_path_RGB, negative_img_path_Depth = self.select_frame_bypath(self.img_dir, negative_path)
            negative_img = cv2.imread(os.path.join(self.img_dir, negative_path, negative_img_path_RGB), cv2.IMREAD_UNCHANGED)
            if self.substract_depth:
                negative_img = substract_depth(cv2.imread(os.path.join(self.img_dir, negative_path, negative_img_path_Depth), cv2.IMREAD_UNCHANGED), negative_img, isTensor=False)


            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            # return les trois imgs et le label de l ancre
            return anchor_img, positive_img, negative_img, idx

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
            return anchor_img, positive_img


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


class LitResNet50(L.LightningModule):
    def __init__(self, embedding_size):
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

        self.loss = torch.jit.script(TripletLoss(margin=embedding_size))

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def freeze_feature_extractor(self, layers_not_freeze = 4):
        # Gel de tous les paramètres du feature_extractor sauf les n derniers layers
        layers = list(self.resnet50.children())  # Obtenir tous les layers du feature_extractor
        layers_to_freeze = layers[:-layers_not_freeze]  # Tous sauf les n derniers layers

        # Geler les paramètres des layers sélectionnés
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Passer l'image à travers ResNet50 (partie feature extraction)
        features = self.resnet50(x)

        # Aplatir les caractéristiques
        features = features.view(features.size(0), -1)

        # Obtenir l'embedding
        embedding = self.embedding_layer(features)

        return embedding

    def training_step(self, batch, batch_idx):

        anchor_img, positive_img, negative_img, idx = batch
        # Obtenir l'embedding à partir de l'image

        anchor_out = self.forward(anchor_img)
        positive_out = self.forward(positive_img)
        negative_out = self.forward(negative_img)

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

        anchor_img, positive_img, negative_img, idx = batch
        # Obtenir l'embedding à partir de l'image

        anchor_out = self.forward(anchor_img)
        positive_out = self.forward(positive_img)
        negative_out = self.forward(negative_img)
        val_loss = self.loss(anchor_out, positive_out, negative_out)

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)

        correct = (d_ap < d_an).sum()
        total = len(d_an)

        if batch_idx == 1: # Log every 10 batches
            self.log_tb_images([anchor_img, positive_img, negative_img, d_ap, d_an])

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
            images_concat = torch.cat((anchor_img, positive_img, negative_img), dim=2)
            tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", images_concat, 0)
            # tb_logger.add_image(f"ImagePositive/{batch_idx}_{img_idx}", positive_img, 0)
            # tb_logger.add_image(f"PositiveDistance/{batch_idx}_{img_idx}", d_ap, 0)
            # tb_logger.add_image(f"ImageNegative/{batch_idx}_{img_idx}", negative_img, 0)
            # tb_logger.add_image(f"NegativeDistance/{batch_idx}_{img_idx}", d_an, 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer
    
    def print_layers(self):
        # Imprimer le nom de la couche et si elle est gelée (requires_grad == False)
        for name, param in self.named_parameters():
            print(f"{name}: {'gelé' if not param.requires_grad else 'non gelé'}")

## ------------------------------------------------------------------
## SEQUENCE Class

class SequenceVRAI(Dataset):
    def __init__(self, img_labels, img_dir, train=True, transform=None, MAX_SEQUENCE_LENGHT = 10):
        self.is_train = train
        self.transform = transform

        self.img_labels = img_labels
        self.index = self.img_labels.index.values
        self.img_dir = img_dir
        self.maxSequenceLenght = MAX_SEQUENCE_LENGHT

    def __len__(self):
        return len(self.img_labels)

    def bgr_to_rgb(self, bgr_img):
        permute = [2, 1, 0]

        rgb_img = bgr_img[permute]
        return rgb_img

    def sequence_preprocess(self, abs_path, passage_path,  direction):

        images = os.listdir(os.path.join(abs_path, passage_path))
        images_depth = [im for im in images if "_D" in im]
 
        list_frames_depth = []
        list_frames_RGB = []
        for nom_image in images_depth:
            img_depth = cv2.imread(os.path.join(abs_path, passage_path, nom_image), cv2.IMREAD_UNCHANGED)
            if pretraitementDB.is_image_valid(img_depth):
                list_frames_depth.append(img_depth)
                nom_image_RGB = nom_image.replace("_D.", "_RGB.")
                img_RGB = cv2.imread(os.path.join(abs_path, passage_path, nom_image_RGB), cv2.IMREAD_UNCHANGED)
                list_frames_RGB.append(img_RGB)

        sum_before = 0
        freq = len(list_frames_depth) // (self.maxSequenceLenght + 1)
        list_position_depth = []
        for i in range(int(self.maxSequenceLenght)):
            sum_before += freq
            list_position_depth.append(round(sum_before))

        assert len(list_position_depth) == 10

        depth_sequence = []
        RGB_sequence = []
        for i in list_position_depth:
            depth_sequence.append(pretraitementDB.pre_processing(list_frames_depth[i], direction))
            RGB_sequence.append(pretraitementDB.pre_processing(list_frames_RGB[i], direction))

        return depth_sequence

    def select_frame_bypath(self, abs_path, passage_path):

        images = os.listdir(os.path.join(abs_path, passage_path))
        images_depth = [im for im in images if "_D" in im]
        max_frame = select_frame(images_depth, os.path.join(abs_path, passage_path))
        max_frame_RGB = max_frame.replace("_D.", "_RGB.")
        # print(max_frame_RGB)
        return max_frame_RGB

    def __getitem__(self, idx):
        anchor_path = self.img_labels.iloc[idx, 1]
        anchor_img_path = self.select_frame_bypath(self.img_dir, anchor_path)
        anchor_img = read_image(os.path.join(self.img_dir, anchor_path, anchor_img_path))
        anchor_img = self.bgr_to_rgb(anchor_img)

        # le positif correspond au OUT de l idx, soit le champ 1
        positive_path = self.img_labels.iloc[idx, 2]
        positive_img_path = self.select_frame_bypath(self.img_dir, positive_path)
        positive_img = read_image(os.path.join(self.img_dir, positive_path, positive_img_path))
        positive_img = self.bgr_to_rgb(positive_img)

        if self.is_train:
            # Peut etre un reshape ????

            # le negatif correspond au OUT d un autre idx, champ 1 egalement
            negative_list = self.index[self.index!=idx]
            negative_item = random.choice(negative_list)
            negative_path = self.img_labels.iloc[negative_item, 2]
            negative_img_path = self.select_frame_bypath(self.img_dir, negative_path)
            negative_img = read_image(os.path.join(self.img_dir, negative_path, negative_img_path))
            negative_img = self.bgr_to_rgb(negative_img)

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            # return les trois imgs et le label de l ancre
            return anchor_img, positive_img, negative_img, idx

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
            return anchor_img, positive_img

## ------------------------------------------------------------------
## Transformers Class

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x) 

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Positional Encoding.

        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class TransformerPredictor(L.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        warmup,
        max_iters,
        dropout=0.0,
        input_dropout=0.0,
    ):
        """TransformerPredictor.

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout), nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError