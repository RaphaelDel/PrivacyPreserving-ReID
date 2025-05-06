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

from pretraitementDB import is_image_valid, pre_processing

MAX_SEQUENCE_LENGHT = 1

class DepthVRAI(Dataset):
    def __init__(self, img_labels, img_dir, train=True, transform=None, normaliseDepth = True, maxSequenceLenght = MAX_SEQUENCE_LENGHT):
        self.is_train = train
        self.transform = transform
        self.normaliseDepth = normaliseDepth

        self.img_labels = img_labels
        self.index = self.img_labels.index.values
        self.img_dir = img_dir
        self.maxSequenceLenght = maxSequenceLenght

    def __len__(self):
        return len(self.img_labels)

    def sequence_depth(self, abs_path, passage_path,  direction):

        images = os.listdir(os.path.join(abs_path, passage_path))
        images_depth = [im for im in images if "_D" in im]

        list_frames = []
        for nom_image in images_depth:
            img = cv2.imread(os.path.join(abs_path, passage_path, nom_image), cv2.IMREAD_UNCHANGED)
            if is_image_valid(img):
                list_frames.append(img)

        sum_before = 0
        freq = len(list_frames) // (self.maxSequenceLenght + 1)
        list_position_depth = []
        for i in range(int(self.maxSequenceLenght)):
            sum_before += freq
            list_position_depth.append(round(sum_before))

        assert len(list_position_depth) == self.maxSequenceLenght

        depth_sequence = []
        for i in list_position_depth:
            depth_sequence.append(pre_processing(list_frames[i], direction, normalise=self.normaliseDepth))

        if self.maxSequenceLenght == 1:
            depth_sequence = depth_sequence[0]

        return depth_sequence

    def __getitem__(self, idx):
        anchor_path = self.img_labels.iloc[idx, 1]
        anchor_depth_sequence = self.sequence_depth(self.img_dir, anchor_path, 0)

        # le positif correspond au OUT de l idx, soit le champ 1
        positive_path = self.img_labels.iloc[idx, 2]
        postive_depth_sequence = self.sequence_depth(self.img_dir, positive_path, 1)

        if self.is_train:
            # Peut etre un reshape ????

            # le negatif correspond au OUT d un autre idx, champ 1 egalement
            negative_list = self.index[self.index!=idx]
            negative_item = random.choice(negative_list)
            negative_path = self.img_labels.iloc[negative_item, 2]
            negative_depth_sequence = self.sequence_depth(self.img_dir, negative_path, 1)

            if self.transform:
                if type(anchor_depth_sequence)  == list:
                    anchor_depth_sequence = torch.stack([self.transform(img) for img in anchor_depth_sequence], 0)
                    postive_depth_sequence = torch.stack([self.transform(img) for img in postive_depth_sequence], 0)
                    negative_depth_sequence = torch.stack([self.transform(img) for img in negative_depth_sequence], 0)
                else:
                    anchor_depth_sequence = self.transform(anchor_depth_sequence)
                    postive_depth_sequence = self.transform(postive_depth_sequence)
                    negative_depth_sequence = self.transform(negative_depth_sequence)
            # return les trois imgs et le label de l ancre
            return anchor_depth_sequence, postive_depth_sequence, negative_depth_sequence, idx

        else:
            if self.transform:
                if type(anchor_depth_sequence)  == list:
                    anchor_depth_sequence = torch.stack([self.transform(img) for img in anchor_depth_sequence], 0)
                    postive_depth_sequence = torch.stack([self.transform(img) for img in postive_depth_sequence], 0)
                else:
                    anchor_depth_sequence = self.transform(anchor_depth_sequence)
                    postive_depth_sequence = self.transform(postive_depth_sequence)
            return anchor_depth_sequence, postive_depth_sequence


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


class SimpleDepthEncoder(L.LightningModule):
    def __init__(self, embedding_size):
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

        self.loss = torch.jit.script(TripletLoss(margin=embedding_size))
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        x = self.conv_layer_2(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.encoder(x)
        return x

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
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer

# --------------------------------------------------------------------
# SEQUENCE DEPTH ENCODER


class GlobalEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout_rate):
        super(GlobalEmbeddingExtractor, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        # Optional: Define a special token for global embedding extraction
        self.global_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, embeddings):
        # Optional: Prepend the global token
        global_token = self.global_token.expand(-1, embeddings.size(1), -1)  # Expand to match the batch size
        embeddings = torch.cat((global_token, embeddings), dim=0)
        
        # Pass embeddings through the transformer
        transformed_embeddings = self.transformer_encoder(embeddings)
        
        # Extract the global embedding (output corresponding to the global token)
        global_embedding = transformed_embeddings[0]  # Assuming the global token is at position 0
        
        return global_embedding

class SequenceDepthEncoder(L.LightningModule):
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


        # Seq with Dense layer
        # self.seqDense = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=embedding_size*NbFrames, out_features=embedding_size)
        # )

        ## Transformer Layers
        # Parameters
        embedding_dim = 256  # Dimensionality of each embedding
        num_heads = 8  # Number of attention heads in the transformer
        num_layers = 2  # Number of transformer layers
        dropout_rate = 0.1  # Dropout rate

        # # Initialize the model
        # self.TransfomerModel = GlobalEmbeddingExtractor(embedding_dim, num_heads, num_layers, dropout_rate)

        # Example input: a batch of sequences, each with 10 embeddings of dimension 256
        # Shape: (batch_size, sequence_length, embedding_dim)

        self.loss = TripletLoss(margin=embedding_size)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image_sequence):
        # From Batch, Seq, Filter, H, W to  : Seq, Batch, Filter, H, W 
        image_sequenceP = image_sequence.permute(1, 0, 2, 3, 4)
        embeddings = []
        for img in image_sequenceP:
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

        # Forward pass to extract global embeddings
        # transformer_output = self.TransfomerModel(embeddings)  # Output shape: (batch_size, embedding_dim)

        # Mean
        transformer_output = embeddings.mean(0)
        
        # Dense fusion information
        # embeddings =  embeddings.permute(1, 0, 2)
        # transformer_output = self.seqDense(embeddings)
        return transformer_output

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