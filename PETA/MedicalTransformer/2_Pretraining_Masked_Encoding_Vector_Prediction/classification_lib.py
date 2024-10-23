import torch.optim as optim
from models.ResNet_Model import Multiview_MEP
from losses import *
from helpers import *
import os
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torchvision
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
import time
from tqdm import tqdm
import pandas as pd
from torchsummary import summary

# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Multiview_Classification(Multiview_MEP): # own
    def __init__(self, args, device):
        super(Multiview_Classification, self).__init__(args, device)

        # num_ftrs = 19584 # (114 + 96 + 96) * 64
        # num_classes = 3
        num_classes = 2
        num_ftrs = (128 + args.axial_slicelen + 128) * 64 # 17408 si #slices = 16

        if args.fc_layers == 2:
            # 1024 is elected in an arbitrary way
            self.fc = nn.Sequential(
                nn.Linear(num_ftrs,1024),
                nn.ReLU(),
                nn.Linear(1024,num_classes),
            )
        elif args.fc_layers == 1:
            self.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise Exception(f"Invalid num of fc layers {args.fc_layers}")

        self.dropout = nn.Dropout(p=args.dropout)
        # self.fc = nn.Linear(num_ftrs, num_classes)

        self.fc.apply(init_normal)
        
    def forward(self, x):
        # encode
        # x [B, 193, 229, 193]
        x = x.unsqueeze(1)  # [B, 1, 193, 229, 193]

        # axial
        x_axial = x.clone()  # [B, 1, 193, 229, 193]
        encoding_axial = self.encoding(x_axial)  # [B, 256, 193]

        emb_axial = encoding_axial.clone()  # [1, 256, 193]

        attn_mask = torch.zeros(encoding_axial.size(0), encoding_axial.size(2))  # [B, 193]
        tf_axial = self.TF(emb_axial.permute(0, 2, 1), attn_mask.to(self.device), [0]).permute(0, 2, 1)  # [B, 256, 193]

        del x_axial
        del encoding_axial
        del emb_axial
        del attn_mask

        torch.cuda.empty_cache() 

        # sagittal
        x_sagittal = x.clone().permute(0, 1, 4, 3, 2)  # [B, 1, 193, 229, 193]
        encoding_sagittal = self.encoding_sag(x_sagittal)  # [B, 256, 193]

        emb_sagittal = encoding_sagittal.clone()  # [1, 256, 193]

        attn_mask = torch.zeros(encoding_sagittal.size(0), encoding_sagittal.size(2))  # [B, 193]
        tf_sagittal = self.TF(emb_sagittal.permute(0, 2, 1), attn_mask.to(self.device), [1]).permute(0, 2, 1)  # [B, 256, 193]

        del x_sagittal
        del encoding_sagittal
        del emb_sagittal
        del attn_mask

        torch.cuda.empty_cache()

        # coronal
        x_coronal = x.clone().permute(0, 1, 2, 4, 3)  # [B, 1, 193, 193, 229]
        encoding_coronal = self.encoding_cor(x_coronal)  # [B, 256, 193]

        emb_coronal = encoding_coronal.clone()  # [1, 256, 193]

        attn_mask = torch.zeros(encoding_coronal.size(0), encoding_coronal.size(2))  # [B, 193]
        tf_coronal = self.TF(emb_coronal.permute(0, 2, 1), attn_mask.to(self.device), [2]).permute(0, 2, 1)  # [B, 256, 193]

        del x_coronal
        del encoding_coronal
        del emb_coronal
        del attn_mask

        torch.cuda.empty_cache() 
        
        # print(tf_axial.shape)    # torch.Size([4, 64, 114])
        # print(tf_sagittal.shape) # torch.Size([4, 64, 96])
        # print(tf_coronal.shape)  # torch.Size([4, 64, 96])
        tensor = torch.cat([tf_axial, tf_sagittal, tf_coronal], 2) # torch.Size([4, 64, 306])
        # print(tensor.shape)

        return self.fc(self.dropout(torch.flatten(tensor, start_dim = 1)))
