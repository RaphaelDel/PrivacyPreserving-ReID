import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset
import torchvision.models as models
import lightning as L
from torchvision import transforms
import os
import random
import cv2
import numpy as np

from pretraitementDB import is_image_valid, pre_processing

MAX_SEQUENCE_LENGHT = 1

class BatchHardVRAI(Dataset):
    def __init__(self, img_labels, img_dir, train=True, transformDepth=None, transformRGB=None,
                normaliseDepth = True, substract_Depth=True, maxSequenceLenght = MAX_SEQUENCE_LENGHT, nb_images_variations = 3):
        self.is_train = train

        self.transformDepth = transformDepth
        if self.transformDepth == None:
            self.transformDepth = transforms.ToTensor()
        self.transformRGB = transformRGB
        if self.transformRGB == None:
            self.transformRGB = transforms.ToTensor()
        self.normaliseDepth = normaliseDepth
        self.substract_Depth = substract_Depth

        self.img_labels = img_labels
        self.index = self.img_labels.index.values
        self.img_dir = img_dir
        self.maxSequenceLenght = maxSequenceLenght
        self.nb_images_variations = nb_images_variations

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
        
        depth_sequence = []
        rgb_sequence = []
        for T in range(self.nb_images_variations):
            # Select a random sample of size self.maxSequenceLenght
            list_position_depth = random.sample(range(len(d_list_frames)), self.maxSequenceLenght)
            list_position_depth.sort()
            assert len(list_position_depth) == self.maxSequenceLenght

            depth_sequence_t = []
            rgb_sequence_t = []
            for i in list_position_depth:
                depthTmpImg = pre_processing(d_list_frames[i], direction, normalise=self.normaliseDepth)
                depth_sequence_t.append(self.transformDepth(depthTmpImg))
                if self.substract_Depth == True:
                    rgbTmpImg  = self.substract_depth(depthTmpImg, pre_processing(rgb_list_frames[i], direction, normalise=False), isTensor=False)
                else:
                    rgbTmpImg  = pre_processing(rgb_list_frames[i], direction, normalise=False)
                rgb_sequence_t.append(self.transformRGB(rgbTmpImg.copy()))

            if self.maxSequenceLenght == 1:
                depth_sequence_t = depth_sequence_t[0]
                rgb_sequence_t = rgb_sequence_t[0]
            
            depth_sequence.append(torch.stack(depth_sequence_t, 0))
            rgb_sequence.append(torch.stack(rgb_sequence_t, 0))

        return torch.stack(depth_sequence, 0), torch.stack(rgb_sequence, 0)

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

            # if self.transformDepth:
                # if type(anchor_depth_sequence)  == list:
                #     anchor_depth_sequence = torch.stack([self.transformDepth(img) for img in anchor_depth_sequence], 0)
                #     postive_depth_sequence = torch.stack([self.transformDepth(img) for img in postive_depth_sequence], 0)
                #     negative_depth_sequence = torch.stack([self.transformDepth(img) for img in negative_depth_sequence], 0)

                #     anchor_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in anchor_rgb_sequence], 0)
                #     postive_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in postive_rgb_sequence], 0)
                #     negative_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in negative_rgb_sequence], 0)
                # else:
                #     anchor_depth_sequence = self.transformDepth(anchor_depth_sequence)
                #     postive_depth_sequence = self.transformDepth(postive_depth_sequence)
                #     negative_depth_sequence = self.transformDepth(negative_depth_sequence)

                #     anchor_rgb_sequence = self.transformRGB(anchor_rgb_sequence)
                #     postive_rgb_sequence = self.transformRGB(postive_rgb_sequence)
                #     negative_rgb_sequence = self.transformRGB(negative_rgb_sequence)
            # return les trois imgs et le label de l ancre
            return (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence), (negative_depth_sequence, negative_rgb_sequence), idx

        else:
            # if self.transformDepth:
            #     if type(anchor_depth_sequence)  == list:
            #         anchor_depth_sequence = torch.stack([self.transformDepth(img) for img in anchor_depth_sequence], 0)
            #         postive_depth_sequence = torch.stack([self.transformDepth(img) for img in postive_depth_sequence], 0)

            #         anchor_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in anchor_rgb_sequence], 0)
            #         postive_rgb_sequence = torch.stack([self.transformRGB(img.copy()) for img in postive_rgb_sequence], 0)
            #     else:
            #         anchor_depth_sequence = self.transformDepth(anchor_depth_sequence)
            #         postive_depth_sequence = self.transformDepth(postive_depth_sequence)

            #         anchor_rgb_sequence = self.transformRGB(anchor_rgb_sequence)
            #         postive_rgb_sequence = self.transformRGB(postive_rgb_sequence)
            return (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence)



class HardBatchedTripletLoss(nn.Module):
    def __init__(self, margin=0.25):
        super(HardBatchedTripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def euclidean_distance_matrix(self, tensor1, tensor2):
        dist_matrix = torch.cdist(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).squeeze(0)
        return dist_matrix.flatten(1)
    
    # Calculer la matrice de distance du cosinus
    def cosine_distance_matrix_batch(self, tensor1, tensor2):
        tensor1_norm = tensor1 / tensor1.norm(dim=2, keepdim=True)
        tensor2_norm = tensor2 / tensor2.norm(dim=2, keepdim=True)
        cosine_similarity = torch.bmm(tensor1_norm, tensor2_norm.transpose(1, 2))
        cosine_distance = 1 - cosine_similarity
        return cosine_distance.flatten(1)


    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # print(f"Inside BatchedTripletLoss : {negative.size()}")
        B, N, hidden_size = negative.size()

        distance_positive, _ = self.euclidean_distance_matrix(anchor, positive).max(1, keepdim=True)
        # print("distance_positive", distance_positive)
        distance_negative, _ = self.euclidean_distance_matrix(anchor, negative.view(-1, hidden_size).expand(B, -1, -1)).min(1, keepdim=True)
        # print("distance_negative", distance_negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        # print("losses", losses)

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
        self.layer_norm = nn.LayerNorm([NbFrames, embedding_size])

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

        output = self.layer_norm(embeddings.permute(1, 0, 2)).permute(1, 0, 2)
        return output

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

        self.layer_norm = nn.LayerNorm([NbFrames, embedding_size])
    
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
            # print("img  size : ", img.size())
            features = self.resnet50(img)
            # print("features  size : ", features.size())
            # Aplatir les caractéristiques
            features = features.view(features.size(0), -1)
            # print("features  size : ", features.size())
            # Obtenir l'embedding
            embedding = self.embedding_layer(features)
            # print("embedding.size()", embedding.size())
            embeddings.append(embedding)
        # print("len(embedding)", len(embeddings))
        embeddings = torch.stack(embeddings)  # Convert list of tensors to a tensor
        # print("embedding.size()", embeddings.size())
        # OutPut size : Seq, Batch, EmbeddSize

        # Mean
        output = self.layer_norm(embeddings.permute(1, 0, 2)).permute(1, 0, 2)
        return output


class LearnedSeqEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=10, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len
        self.device = device
        self.learned_encoding = nn.Embedding(self.seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # print("learned_encoding size", self.learned_encoding(torch.arange(self.seq_len).to(self.device)).size())
        # print("learned_encoding size", self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).size())

        x = x + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1)

        return self.dropout(x)


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(TransformerEncoder, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class SequenceEncoder(L.LightningModule):
    def __init__(self, embedding_size, NbFrames, Nb_layerToFreeze=4):
        super().__init__()

        # Depth Sequence Encoder
        self.DepthSeqEncoder = SequenceDepthEncoder(embedding_size, NbFrames)
        self.RGBSeqEncoder = SequenceRGBEncoder(embedding_size, NbFrames)
        self.RGBSeqEncoder.freeze_feature_extractor(Nb_layerToFreeze)

        self.positional_encoder = LearnedSeqEncoding(d_model = 2*embedding_size, seq_len=NbFrames)
        encoder_layer_local = nn.TransformerEncoderLayer(d_model=2*embedding_size,
                                                   nhead=8,
                                                   dim_feedforward=1024,
                                                   dropout=0.1,
                                                   activation='relu',
                                                   batch_first=False)
        self.local_former = TransformerEncoder(encoder_layer_local, num_layers = 6)
        # self.fc_out = nn.Sequential(nn.Linear(2*embedding_size, 1024),
        #                             nn.ReLU(),
        #                             nn.Linear(1024, embedding_size),
        #                             nn.ReLU())
                                    

        self.loss = HardBatchedTripletLoss(margin=1.0)
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Optimizer
        self.warmup_steps = 10
        self.learning_rate = 3.5e-4

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
        
        #Transformer part
        # print(" size before cat", depth_encoded.size(), rgb_encoded.size())
        tinput = torch.cat((depth_encoded, rgb_encoded), -1)
        # print("tinput size", tinput.size())

        tinput = self.positional_encoder(tinput)
        # print("tinput size", tinput.size())
        out_local = self.local_former(tinput)
        # print("out_local size", out_local.size())

        #Residual connexion 
        #TODO test avec ou sans
        ratio = 0.5
        out_local = (out_local * ratio) + (tinput * (1-ratio))

        #TODO prend uniquement l element 0 en sortie
        # a voir si c est optimal, peut etre un moy ou une cat
        # output = self.fc_out(out_local[0])
        output = torch.mean(out_local, dim=0)
        # output = self.fc_out(output)
        # print("output size", output.size())
        return output, torch.mean(depth_encoded, dim=0), torch.mean(rgb_encoded, dim=0)

    def training_step(self, batch, batch_idx):

        (anchor_depth_sequence, anchor_rgb_sequence), (postive_depth_sequence, postive_rgb_sequence), (negative_depth_sequence, negative_rgb_sequence), idx = batch
        # Obtenir l'embedding à partir de l'image
        # print("anchor_depth_sequence.size()", anchor_depth_sequence.size())
        B, T, N, C, H, W = anchor_depth_sequence.size()

        anchor_out, depth_anchor, rgb_anchor = self.forward(anchor_depth_sequence.view(-1, N, 1, H, W), anchor_rgb_sequence.view(-1, N, 3, H, W))
        positive_out, depth_positive, rgb_positive = self.forward(postive_depth_sequence.view(-1, N, 1, H, W), postive_rgb_sequence.view(-1, N, 3, H, W))
        negative_out, depth_negative, rgb_negative = self.forward(negative_depth_sequence.view(-1, N, 1, H, W), negative_rgb_sequence.view(-1, N, 3, H, W))
        # print("anchor_out.size(), depth_anchor.size(), rgb_anchor.size() ", anchor_out.size(), depth_anchor.size(), rgb_anchor.size())

        train_loss = self.loss(anchor_out.view(B, T, -1), positive_out.view(B, T, -1), negative_out.view(B, T, -1))
        depth_loss = self.loss(depth_anchor.view(B, T, -1), depth_positive.view(B, T, -1), depth_negative.view(B, T, -1))
        rgb_loss = self.loss(rgb_anchor.view(B, T, -1), rgb_positive.view(B, T, -1), rgb_negative.view(B, T, -1))
        total_loss = train_loss + (depth_loss + rgb_loss)

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)

        correct = (d_ap < d_an).sum()
        total = len(d_an)

        self.log("Train/train_loss", train_loss, prog_bar=True)
        self.log("Train/depth_loss", depth_loss, prog_bar=True)
        self.log("Train/rgb_loss", rgb_loss, prog_bar=True)
        self.log("Train/Total_loss", total_loss, prog_bar=True)
        self.training_step_outputs.append({"loss" : train_loss, "correct" : correct, "total" : total})
        return total_loss

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
        B, T, N, C, H, W = anchor_depth_sequence.size()

        anchor_out, depth_anchor, rgb_anchor = self.forward(anchor_depth_sequence.view(-1, N, 1, H, W), anchor_rgb_sequence.view(-1, N, 3, H, W))
        positive_out, depth_positive, rgb_positive = self.forward(postive_depth_sequence.view(-1, N, 1, H, W), postive_rgb_sequence.view(-1, N, 3, H, W))
        negative_out, depth_negative, rgb_negative = self.forward(negative_depth_sequence.view(-1, N, 1, H, W), negative_rgb_sequence.view(-1, N, 3, H, W))

        val_loss = self.loss(anchor_out.view(B, T, -1), positive_out.view(B, T, -1), negative_out.view(B, T, -1))

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)

        correct = (d_ap < d_an).sum()
        total = len(d_an)

        # if batch_idx == 1: # Log every 10 batches
        #     self.log_tb_images([anchor_depth_sequence, postive_depth_sequence, negative_depth_sequence, d_ap, d_an])

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
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Lambda function for the learning rate schedule with warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            elif current_step >= self.warmup_steps and current_step < 40:
                return float(1.0)
            elif current_step >= 40 and current_step < 70:
                return float(0.1)
            elif current_step >= 70 and current_step < 100:
                return float(0.01)
            return max(
                0.0, float(1.0) / float(current_step)
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(scheduler.get_last_lr())

        return [optimizer], [scheduler]