import torch.optim as optim
# from models.ResNet_Model import Multiview_MEP
from models.ResNet_Model import Plane_Feature_extraction
#from models.ResNet import generate_model as generate_model_3d
#from models.ResNet_2d import resnet2d as resnet2d
from opts import parse_opts
from losses import *
from helpers import *
import os
import random
import datetime
import time
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
import torchvision
from torchvision import transforms
# from config import models_genesis_config
from tqdm import tqdm
import pandas as pd
import wandb

import sys
sys.path.append('../../src')
from transforms import ToLabelOutput, ToLabelOutputConfigurable, TransformGridImage, TransformReduced3DImage, Transform3DImage, MinMaxNormalization
from datasets import ADNIDataset
from datasets_pretrain import PretrainDatasetBuilder
from dl_builder import DLBuilder

SETS_FOLDER = "../../Sets"
IMAGES_ROOT_FOLDER = "../../.."

# Define Arguments
args = parse_opts()

# GPU Configuration
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
print(f"Device: cuda:{gpu_id}")
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Load Dataset
# images = np.load('/home/eiarussi/Proyectos/Fleni/MedicalTransformer/images.npy', mmap_mode="r", allow_pickle=True)  # (566, 193, 229, 193)
# indexes = np.load('/home/eiarussi/Proyectos/Fleni/MedicalTransformer/indices.npy', allow_pickle=True)  # (509,), (57,)

# ixi_images = np.load('/DataCommon/ejjun/dataset/IXI/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (566, 193, 229, 193)
# camcan_images = np.load('/DataCommon/ejjun/dataset/camcan/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (653, 193, 229, 193)
# abide_images = np.load('/DataCommon/ejjun/dataset/ABIDE/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (564, 193, 229, 193)
# images = np.concatenate((ixi_images, camcan_images, abide_images), axis=0)
# np.save('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/data_norm_min_max.npy', images)
labels = []
#o ixi_indexes = np.load('/DataCommon/ejjun/dataset/IXI/output/10idx.npy', allow_pickle=True)  # (509,), (57,)
# camcan_indexes = np.load('/DataCommon/ejjun/dataset/camcan/output/10idx.npy', allow_pickle=True)  # (587,), (66,)
# abide_indexes = np.load('/DataCommon/ejjun/dataset/ABIDE/output/10idx.npy', allow_pickle=True)  # (507,), (57,)
#
# indexes = [np.concatenate((ixi_indexes[0], camcan_indexes[0]+np.array(len(ixi_images)), abide_indexes[0]+np.array(len(ixi_images))+np.array(len(camcan_images))), axis=0),
#            np.concatenate((ixi_indexes[1], camcan_indexes[1]+np.array(len(ixi_images)), abide_indexes[1]+np.array(len(ixi_images))+np.array(len(camcan_images))), axis=0)]
# np.save('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/10idx.npy', indexes)
# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))

if args.name:
    directory = './log/%s/' % (args.name)
else:
    directory = './log/%s/%s/%s_batch_%d_lr_%f_lambda2_%f_ResNet_%d_inplanes_%d/' % (args.pre_dataset, args.approach, date_str, args.batch_size, args.lr, args.lambda2, args.depth, args.inplanes)

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + 'img/')
    os.makedirs(directory + 'img/train/')
    os.makedirs(directory + 'img/valid/')
    os.makedirs(directory + 'img/test/')
    os.makedirs(directory + 'tflog/')
    os.makedirs(directory + 'model/')

# Text Logging
f = open(directory + 'setting.log', 'a')
writelog(f, '======================')
# writelog(f, 'Model: %s' % (args.model))
writelog(f, 'Log directory: %s' % directory)
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, 'LRSched Gamma: %.5f' % args.steplr_gamma)
writelog(f, 'LRSched Step Size: %.5f' % args.steplr_step_size)
writelog(f, 'LRSched Last epoch: %.5f' % args.steplr_last_epoch)
writelog(f, 'Nonlinear prob: %.2f' % args.nonlinear_prob)
writelog(f, '----------------------')
writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.8f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Eval rounds: %d' % args.eval_rounds)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '======================')
f.close()

if args.wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="medicaltransformer-cnn",
        entity="hugomassaroli",
        name = args.name,
        # track hyperparameters and run metadata
        config={
            "step": "prepretrain",
            "learning_rate": args.lr,
            "seed": args.seed,
            "augmentation": args.augmentation,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "lambda2": args.lambda2,
            "steplr_step_size": args.steplr_step_size,
            "steplr_gamma": args.steplr_gamma,
            "steplr_last_epoch": args.steplr_last_epoch,
            "depth": args.depth,
            "inplanes": args.inplanes,
            "d_f": args.d_f,
            "max_slicelen": args.max_slicelen,
            "axial_slicelen": args.axial_slicelen,
            "coronal_slicelen": args.coronal_slicelen,
            "dataset": args.dataset,
            "eval_dataset": args.eval_dataset,
            "eval": args.eval,
            "subset_size": args.subset_size,
            "miniset_size": args.miniset_size,
            "weights": args.weights,
            "eval_set_size": args.eval_set_size,
            "eval_shuffle": args.eval_shuffle,
            "eval_rounds": args.eval_rounds,
            "nonlinear_prob": args.nonlinear_prob,
            "truth_label": args.truth_label,
        }
    )

f = open(directory + 'log.log', 'a')
# Tensorboard Logging
# tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/train_')
# tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/valid_')
# tfw_test = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/test_')
tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog/train_')
tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog/valid_')

# Tensor Seed
if args.seed > 0:
    writelog(f, f"Using seed = {args.seed}")
    random_state = args.seed # args.seed 123456
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
else:
    writelog(f, "No seed selected")
    random_state = None

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

num_workers = args.num_workers

writelog(f, 'Load training data')
imagesDir = "/home/eiarussi/Proyectos/Fleni/ADNI-MUESTRA-FULL-stripped-preprocessed3/"
#train_loader = sample_loader('train', images, labels, indexes, args, drop_last=True)
dataLoaderArgs = {
    "batch_size": args.batch_size,
    "num_workers": num_workers,
    "shuffle": False, # falso porque usa el PretrainDatasetBuilder TODO: volver a setear
    "pin_memory": True,
    "drop_last": False
}

if args.normalization == "min-max":
    normalization = MinMaxNormalization(args.min_intensity, 1)
    writelog(f, f"Normalization: MinMax({args.min_intensity}, 1)")
elif args.normalization == "z-score":
    trainMean = 0.26102542877197266
    trainStd = 0.46347398656747363
    normalization = transforms.Normalize([trainMean], [trainStd])
    writelog(f, f"Normalization: Z-Score({trainMean}, {trainStd})")
else:
    raise Exception(f"Unknown normalization {args.normalization}")
    
if args.axial_slicelen == 16:
    writelog(f, "Using TransformReduced3DImage")
    # transform3D = TransformReduced3DImage()
    valTransform3D = TransformReduced3DImage()
else:
    writelog(f, f"Using Transform3DImage with slicelen = {args.axial_slicelen}")
    # transform3D = Transform3DImage(yDim = args.axial_slicelen, augmentation = args.augmentation, **dataAugmentation)
    valTransform3D = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no')

valTransform = torchvision.transforms.Compose([
    valTransform3D,
    torchvision.transforms.ToTensor(),
    normalization
])

if args.augmentation in ['all', 'one']:
    if args.da_yiming:
        writelog(f, "Using same Da as Yiming Ding for experiment compatibility")
        dataAugmentation = {
        "angle": 8,
        "shiftX": 10,
        "shiftY": 10,
        "zoom": 0.1,
        "shear": 0.19634954084936207
        }
    else:
        dataAugmentation = {
            "angle": 15,
            "shiftX": 10,
            "shiftY": 10,
            "zoom": 0.12,
            "shear": 0.19634954084936207
        }
        if args.collapse_ydim > 0.0:
            dataAugmentation["collapseYDim"] = 47
            dataAugmentation["collapseYDimChance"] = args.collapse_ydim
else:
    dataAugmentation = {}

writelog(f, f"Augmentation: {args.augmentation}")
writelog(f, str(dataAugmentation))


pretrainADNIMini = False
pretrainFleni = False
pretrainADNI = False
pretrainMerida = False
pretrainChinese = False

if args.dataset == 'mixed':
    pretrainADNI = True
    pretrainFleni = True
elif args.dataset == 'adni-plus':
    pretrainADNI = True
    pretrainMerida = True
    pretrainChinese = True
elif args.dataset == 'all':
    pretrainADNI = True
    pretrainFleni = True
    pretrainMerida = True
    pretrainChinese = True
elif args.dataset == 'adni':
    pretrainADNI = True
elif args.dataset == 'fleni':
    pretrainFleni = True
elif args.dataset == 'mini':
    pretrainADNIMini = True
else:
    raise Exception(f"Dataset not implemented = {args.dataset}")

writelog(f, f"Dataset: {args.dataset}")

# Checking truth label
if args.truth_label == "last":
    truthLabel = "DX_last"
    trainDatasetCSV = "Muestra3700_80_10_10_dxlast_train.csv"
    valDatasetCSV = "Muestra3700_80_10_10_dxlast_val.csv"
    testDatasetCSV = "Muestra3700_80_10_10_dxlast_test.csv"
elif args.truth_label == "visit":
    truthLabel = "DX_vis"
    trainDatasetCSV = "Muestra3700_80_10_10_dxvisit953_train.csv"
    valDatasetCSV = "Muestra3700_80_10_10_dxvisit953_val.csv"
    testDatasetCSV = "Muestra3700_80_10_10_dxvisit953_test.csv"
else:
    raise Exception(f"Not supported truth label = {args.truth_label}")
writelog(f, f"Truth label: {args.truth_label}")

pretrainConfiguration = {
    "adniMini": pretrainADNIMini,
    "adni": pretrainADNI,
    "fleni": pretrainFleni,
    "merida": pretrainMerida,
    "chinese": pretrainChinese,
    "subset_size": args.subset_size,  # fleni 1009 para fleni + adni en igualdad
    # "subset_size": 3033, # usamos el mas grande (el de ADNI), esto quiere decir que usará aprox 6k por cada epoch
    # "subset_size": 3033//2, # mismo número de muestras que Experimento 1, pero realmente serían mitad adni y mitad fleni
    "num_classes": 2,
    "axial_slicelen": args.axial_slicelen,
    "miniset_size": args.miniset_size,
    "replacement": args.with_replacement,
    "adniTruthLabel": truthLabel,
}
writelog(f, "Pretrain configuration:")
writelog(f, str(pretrainConfiguration))

train_dataset, train_loader = PretrainDatasetBuilder(dataAugmentation, dataLoaderArgs, normalization, random_state = random_state, augmentation = args.augmentation, adniDatasetCSV = f"../../Sets/{trainDatasetCSV}", **pretrainConfiguration).build()

writelog(f, f"Train samples: {len(train_dataset)}")
writelog(f, str(pd.read_csv(f"../../Sets/{trainDatasetCSV}")))

# ----



# trainTransform = torchvision.transforms.Compose([
#     TransformReduced3DImage(**dataAugmentation),
#     torchvision.transforms.ToTensor(),
#     normalization
# ])

# # train_dataset = ADNIDataset('train', '../../Sets/Muestra3700_80_10_10_dxlast_train.csv', imagesDir, transform = trainTransform, target_transform = ToLabelOutput(), truthLabel = "DX_last")
# train_dataset = ADNIDataset('train', '../../Sets/Muestra3700_80_10_10_dxlast_train.csv', imagesDir, transform = trainTransform, target_transform = ToLabelOutput(), truthLabel = "DX_last") # mini
# writelog(f, f"Train samples: {len(train_dataset)}")
drop_last = False
# train_loader = DataLoader(train_dataset,
#                         batch_size=args.batch_size,
#                         num_workers=num_workers,
#                         shuffle=True,
#                         pin_memory=True,
#                         drop_last=drop_last)

#writelog(f, f"Weights for train loader: {train_loader.weights.shape}")
#writelog(f, f"Weights for train loader: {train_loader.weights}")

writelog(f, 'Load validation data')
# valid_loader = sample_loader('valid', images, labels, indexes, args, shuffle=False)
#valid_loader = sample_loader('valid', images, labels, indexes, args, shuffle=False, drop_last=True)
if args.eval_dataset == 'adni':
    writelog(f, "Using full ADNI dataset as eval")
    val_csv = f"../../Sets/{valDatasetCSV}"
    writelog(f, f"Using {valDatasetCSV}")
elif args.eval_dataset == 'adni-test':
    writelog(f, "Using full ADNI TEST dataset as eval")
    val_csv = f"../../Sets/{testDatasetCSV}"
    writelog(f, f"Using {testDatasetCSV}")
elif args.eval_dataset == 'adni_train':
    # Por si queremos testear consistencia
    writelog(f, "Using full ADNI TRAIN dataset as eval")
    val_csv = f"../../Sets/{trainDatasetCSV}.csv"
    val_csv = pd.read_csv(val_csv)
    if args.miniset_size != None:
        val_csv = val_csv.sample(n=args.miniset_size, random_state = random_state)
        writelog(f, f"Reducing adni VAL set to fixed n = {args.miniset_size} and random_state = {random_state}")
        writelog(f, "Selected samples:")
        writelog(f, str(val_csv['Image Data ID']))
elif args.eval_dataset == 'mini':
    if args.truth_label != "last":
        raise Exception("Mini dataset not available for this truth label")
    writelog(f, "Using mini eval dataset")
    val_csv = '../../Sets/Muestra3700_80_10_10_dxlast_val_mini.csv'
elif args.eval_dataset == 'mini2':
    if args.truth_label != "last":
        raise Exception("Mini dataset not available for this truth label")
    writelog(f, "Using mini eval dataset")
    val_csv = '../../Sets/Muestra3700_80_10_10_dxlast_val_mini2.csv'
else:
    raise Exception(f"eval dataset desconocido: {args.eval_dataset}")

if args.eval_set_size != None:
    writelog(f, f"eval_frac != 1.0. Takin samples of n = {args.eval_set_size} for val")
    val_csv = pd.read_csv(val_csv)
    val_csv = val_csv.sample(n=args.eval_set_size, random_state = random_state)

valid_dataset = ADNIDataset('valid', val_csv, imagesDir, transform = valTransform, target_transform = ToLabelOutput(), truthLabel = truthLabel)
writelog(f, f"Shuffle eval: {args.eval_shuffle}")
valid_loader = DataLoader(valid_dataset,
                        batch_size=args.batch_size,
                        num_workers=num_workers,
                        shuffle= args.eval_shuffle == 1,
                        pin_memory=True,
                        drop_last=drop_last)

writelog(f, f"Validation samples: {len(valid_dataset)}")
if isinstance(val_csv,str):
    writelog(f, str(pd.read_csv(val_csv)))
else:
    writelog(f, str(val_csv))

# writelog(f, 'Load test data')
# test_loader = sample_loader('test', images, labels, indexes, args, shuffle=False)
dataloaders = {'train': train_loader,
               'valid': valid_loader}
               # 'test': test_loader}

if args.eval_fleni:
    dlArgs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": drop_last
    }

    if args.axial_slicelen == 16:
        fleniTransform3D = TransformReduced3DImage()
    else:
        fleniTransform3D = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no', resampleZ = 0.610389)
        
    dlFleniBuilder = DLBuilder(SETS_FOLDER, IMAGES_ROOT_FOLDER, dlArgs, normalization, gridTransform = fleniTransform3D)

    dataloaders['fleni100'] = dlFleniBuilder.fleni100() 
    dataloaders['fleni60'] = dlFleniBuilder.fleni60()

if args.approach == '25d':
    model = nn.DataParallel(Plane_Feature_extraction(args)).to(device)
elif args.approach == '2d':
    model = nn.DataParallel(resnet2d(args)).to(device)
elif args.approach == '3d':
    model = nn.DataParallel(generate_model_3d(model_depth=args.model_depth,
                                              inplanes=args.inplanes,
                                              n_classes=args.d_f,
                                              n_input_channels=1,
                                              shortcut_type=args.resnet_shortcut,
                                              conv1_t_size=args.conv1_t_size,
                                              conv1_t_stride=args.conv1_t_stride,
                                              no_max_pool=args.no_max_pool,
                                              widen_factor=args.resnet_widen_factor)).to(device)
else:
    pass

if args.weights != None:
    writelog(f, f"Loading weights from {args.weights}")
    model.load_state_dict(torch.load(args.weights))
             

criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr_step_size, gamma=args.steplr_gamma, last_epoch=args.steplr_last_epoch)

# Define data type
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Best epoch checking
valid = {
    'epoch': 0,
    'auc': 0,
}


def train(dataloader, dir='.'):
    model.train()

    # Define training variables
    train_loss = 0
    n_batches = 0

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):
        inputs = xbatch[0]
        labels = xbatch[1]
        x = Variable(inputs.float(), requires_grad=True).cuda()  # [B, 193, 229, 193]

        optimizer.zero_grad()

        if args.approach == '25d':
            # anchor
            a_cor = x.clone().permute(0, 1, 3, 2)  # [B, 193, 193, 229]
            a_sag = x.clone().permute(0, 3, 2, 1)  # [B, 193, 229, 193]
            a_axial = x.clone()  # [B, 193, 229, 193]

            # positive examples
            p_cor = nonlinear_transformation(a_cor.cpu().detach().numpy(), args.nonlinear_prob)
            p_sag = nonlinear_transformation(a_sag.cpu().detach().numpy(), args.nonlinear_prob)
            p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy(), args.nonlinear_prob)

            # negative examples
            n_cor = torch.cat([a_cor[1:], a_cor[0].unsqueeze(0)], dim=0)
            n_sag = torch.cat([a_sag[1:], a_sag[0].unsqueeze(0)], dim=0)
            n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)

            # anchor
            encoded_a_cor = model(a_cor, plane='cor')
            encoded_a_sag = model(a_sag, plane='sag')
            encoded_a_axial = model(a_axial, plane='axial')

            # positive examples
            encoded_p_cor = model(torch.from_numpy(p_cor).float().cuda(), plane='cor')
            encoded_p_sag = model(torch.from_numpy(p_sag).float().cuda(), plane='sag')
            encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda(), plane='axial')

            # negative examples
            encoded_n_cor = model(n_cor, plane='cor')
            encoded_n_sag = model(n_sag, plane='sag')
            encoded_n_axial = model(n_axial, plane='axial')

            loss_cor = criterion(encoded_a_cor, encoded_p_cor, encoded_n_cor)
            loss_sag = criterion(encoded_a_sag, encoded_p_sag, encoded_n_sag)
            loss_axial = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
            loss = loss_cor + loss_sag + loss_axial

        elif args.approach == '2d':
            # anchor
            a_axial = x  # [2, 96, 114, 96]

            # positive examples
            p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

            # negative examples
            n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

            encoded_a_axial = model(a_axial)  # [2, 64]
            encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
            encoded_n_axial = model(n_axial)

            loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)

        elif args.approach == '3d':
            # anchor
            a_axial = x  # [2, 96, 114, 96]

            # positive examples
            p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

            # negative examples
            n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

            encoded_a_axial = model(a_axial)  # [2, 64]
            encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
            encoded_n_axial = model(n_axial)

            loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
        else:
            pass

        loss.backward()
        optimizer.step()

        print('Training loss = (%.5f)' % loss)
        train_loss += (loss.item() * x.size(0))
        n_batches += 1

    # Take average
    # train_loss = train_loss / n_batches
    if args.subset_size:
        train_loss = train_loss / args.subset_size
    else:
        train_loss = train_loss / len(dataloader.dataset)
    writelog(f, 'Train Loss: %.8f' % train_loss)

    # Tensorboard Logging
    info = {
            'loss': train_loss,
           }
    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        tfw_train.add_summary(summary, epoch)

    return train_loss

def matrix_distance(A, B, dim = (1,2)):
    # Calculate the squared L2 distance element-wise between the matrices
    squared_distance = torch.sum((A - B) ** 2, dim=dim)
    
    # Take the square root to get the L2 (Euclidean) distance
    distance = torch.sqrt(squared_distance)
    
    return distance

        
# útil para evaluar varias rondas
def evaluate_rounds(phase, dataloader, dir='.', rounds = 1):
    losses = []
    accs = []
    for i in range(0, rounds):
        writelog(f, f"Evaluation round {i}")
        round_loss, acc = evaluate(phase, dataloader, dir)
        losses.append(round_loss)
        accs.append(acc)
    losses = torch.Tensor(losses)
    accs = torch.Tensor(accs)
    # returns average of all loss
    loss = losses.sum() / rounds
    writelog(f, f"Averaged Validation Loss: {loss}")
    acc = accs.sum() / rounds
    writelog(f, f"Averaged Validation Accuracy: {acc}")
    std = torch.std(losses)
    writelog(f, f"Std Validation Loss: {std}")
    std_acc = torch.std(accs)
    writelog(f, f"Std Acc: {std}")
    return loss, std, acc, std_acc

def evaluate(phase, dataloader, dir='.'):
    # Set mode as training
    model.eval()

    # Define training variables
    test_loss = 0
    accuracy = 0
    n_batches = 0

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):
            # x = xbatch['data'].float().cuda()
            inputs = xbatch[0]
            labels = xbatch[1]
            # print("SSHAPE")
            # print(inputs.shape)
            x = Variable(inputs.float(), requires_grad=False).cuda()  # [B, 193, 229, 193]

            if args.approach == '25d':
                # anchor
                a_cor = x.clone().permute(0, 1, 3, 2)  # [B, 193, 193, 229]
                a_sag = x.clone().permute(0, 3, 2, 1)  # [B, 193, 229, 193]
                a_axial = x.clone()  # [B, 193, 229, 193]

                # positive examples
                p_cor = nonlinear_transformation(a_cor.cpu().detach().numpy(), args.nonlinear_prob)
                p_sag = nonlinear_transformation(a_sag.cpu().detach().numpy(), args.nonlinear_prob)
                p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy(), args.nonlinear_prob)

                # negative examples
                n_cor = torch.cat([a_cor[1:], a_cor[0].unsqueeze(0)], dim=0)
                n_sag = torch.cat([a_sag[1:], a_sag[0].unsqueeze(0)], dim=0)
                n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)

                # anchor
                encoded_a_cor = model(a_cor, plane='cor')
                encoded_a_sag = model(a_sag, plane='sag')
                encoded_a_axial = model(a_axial, plane='axial')

                # positive examples
                encoded_p_cor = model(torch.from_numpy(p_cor).float().cuda(), plane='cor')
                encoded_p_sag = model(torch.from_numpy(p_sag).float().cuda(), plane='sag')
                encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda(), plane='axial')

                # negative examples
                encoded_n_cor = model(n_cor, plane='cor')
                encoded_n_sag = model(n_sag, plane='sag')
                encoded_n_axial = model(n_axial, plane='axial')

                loss_cor = criterion(encoded_a_cor, encoded_p_cor, encoded_n_cor)
                loss_sag = criterion(encoded_a_sag, encoded_p_sag, encoded_n_sag)
                loss_axial = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
                loss = loss_cor + loss_sag + loss_axial

                
                # print(encoded_p_cor.shape)
                # print(encoded_p_sag.shape)
                # print(encoded_p_axial.shape)

                # TODO: sería mejor hacer la distancia entre matrices que ya contengan las 3 dimensiones, en vez de hacer average
                # accuracy metric
                distance_pos_cor = matrix_distance(encoded_a_cor, encoded_p_cor)
                distance_neg_cor = matrix_distance(encoded_a_cor, encoded_n_cor)
                distance_pos_sag = matrix_distance(encoded_a_sag, encoded_p_sag)
                distance_neg_sag = matrix_distance(encoded_a_sag, encoded_n_sag)
                distance_pos_axial = matrix_distance(encoded_a_axial, encoded_p_axial)
                distance_neg_axial = matrix_distance(encoded_a_axial, encoded_n_axial)

                # print("distances cor")
                # print(distance_pos_cor.shape)
                # print(distance_pos_cor)
                # print(distance_neg_cor)

                # print("distances sag")
                # print(distance_pos_sag.shape)
                # print(distance_pos_sag)
                # print(distance_neg_sag)

                # print("distances axial")
                # print(distance_pos_axial.shape)
                # print(distance_pos_axial)
                # print(distance_neg_axial)

                # print("Acc")
                distance_pos = (distance_pos_cor + distance_pos_sag + distance_pos_axial)
                distance_neg = (distance_pos_cor + distance_neg_sag + distance_neg_axial)
                # print("Sum of distances (distance_neg - distance_pos):")
                # print(distance_neg - distance_pos)
                minibatch_accuracy = (distance_neg - distance_pos) > 0
                # print(minibatch_accuracy)
                accuracy = accuracy +  minibatch_accuracy.sum()
                # print(accuracy)

                # accuracy = (accuracy / inputs.shape[0]).item()
                # print("acc of minibarch")
                # print(accuracy)
                

            elif args.approach == '2d':
                # anchor
                a_axial = x  # [2, 96, 114, 96]

                # positive examples
                p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

                # negative examples
                n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

                encoded_a_axial = model(a_axial)  # [2, 64]
                encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
                encoded_n_axial = model(n_axial)

                loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)

            elif args.approach == '3d':
                # anchor
                a_axial = x  # [2, 96, 114, 96]

                # positive examples
                p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

                # negative examples
                n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

                encoded_a_axial = model(a_axial)  # [2, 64]
                encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
                encoded_n_axial = model(n_axial)

                loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
            else:
                pass

            print('(%s) loss = (%.5f)' % (phase, loss))
            test_loss += (loss.item() * x.size(0))
            n_batches += 1

    # Take average
    # test_loss = test_loss / n_batches
    test_loss = test_loss / len(dataloader.dataset)
    writelog(f, '%s Loss: %.8f' % (phase, test_loss))
    # print(accuracy)
    accuracy = accuracy / len(dataloader.dataset)
    writelog(f, '%s Accuracy: %.8f' % (phase, accuracy))

    # Tensorboard Logging
    info = {'loss': test_loss}

    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        if phase == 'Validation':
            tfw_valid.add_summary(summary, epoch)
        # else:
        #     tfw_test.add_summary(summary, epoch)

    return test_loss, accuracy

if args.eval != None:
    writelog(f, f"Evaluating...")
    writelog(f, f"Eval number: {args.eval_set_size}")
    epoch = 0 # para que no explote abajo
    # loss_val = evaluate('Validation', dataloaders['valid'], dir=directory)
    loss_val, std, acc, _ = evaluate_rounds('Validation', dataloaders['valid'], dir=directory, rounds = args.eval_rounds)
    writelog(f, f"Validation Loss: {loss_val}")
    
    f.close()
    sys.exit(0)

start_time = time.time()
    
# Train Epoch
ES = EarlyStopping(delta=0, patience=args.patience, verbose=True)
for epoch in range(args.epoch):
    start_epoch_time = time.time()
    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, "lr = %f" % scheduler.get_lr()[0])
    writelog(f, 'Training')
    loss_train = train(dataloaders['train'], dir=directory)

    writelog(f, 'Validation')
    # loss_val = evaluate('Validation', dataloaders['valid'], dir=directory)

    loss_val, std, acc, std_acc = evaluate_rounds('Validation', dataloaders['valid'], dir=directory, rounds = args.eval_rounds)

    # loss_val2 = evaluate('Validation', dataloaders['valid'], dir=directory)

    # writelog(f, f"Compare loss 1 = {loss_val} with loss 2 = {loss_val2}")
    
    if epoch == 0:
        valid['loss'] = loss_val

    # Saving model very X epochs
    if args.save_every != -1:
        if epoch % args.save_every == 0:
            weightsFile = directory + f"/model/prepretrain_model_epoch_{epoch}.pt"
            writelog(f, "Saving model: %s" % weightsFile)
            torch.save(model.state_dict(), weightsFile)

    # Save Model
    if loss_val < valid['loss']:
        weightsFile = directory + '/model/prepretrain_model.pt'
        if args.save_best == 1:
            torch.save(model.state_dict(), weightsFile)
            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict()
            # }, os.path.join(directory + '/model', args.model + '_' + str(epoch) + '.pt'))
            writelog(f, "Saving model: %s" % weightsFile)

        writelog(f, 'Best validation loss is found! Validation loss : %f' % loss_val)
        if args.save_best:
            writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss_val
        valid['epoch'] = epoch

    if args.eval_fleni:
        loss_fleni100, acc_fleni100 = evaluate('Fleni100', dataloaders['fleni100'], dir=directory)
        loss_fleni60, acc_fleni60 = evaluate('Fleni60', dataloaders['fleni60'], dir=directory)

        if args.wandb:
            wandb.log({"fleni100_loss": loss_fleni100, "fleni60_loss": loss_fleni60, "fleni100_acc": acc_fleni100, "fleni60_acc": acc_fleni60 }, commit = False)

    total_epoch_time = time.time() - start_epoch_time
    writelog(f, "Epoch time: %.2f m" % (total_epoch_time / 60))
    writelog(f, "Remaining expected time: %.2f hs" % (total_epoch_time * (args.epoch - epoch)/ 60 / 60))

    if args.wandb:
        # log metrics to wandb
        wandb.log({"train_loss": loss_train, "avg_acc": acc, "avg_loss": loss_val, "std": std, "std_acc": std_acc, "min_avg_val_loss": valid['loss'], "remaining_time": total_epoch_time * (args.epoch - epoch)/ 60 / 60})
        
    ES(loss_val, None)

    # loss_test = evaluate('Test', dataloaders['test'], dir=directory)

    scheduler.step()
    if ES.early_stop == True:
        break

end_time = time.time()
total_time = end_time - start_time
avg_time_per_epoch = total_time / args.epoch
writelog(f, "Total time: %.2f hs" % (total_time/60/60))
writelog(f, "Avg time per epoch: %.2f m" %  (avg_time_per_epoch/60))
    
if args.wandb:
    wandb.log({"time_per_epoch": avg_time_per_epoch, "total_train_time": total_time})
    wandb.finish()
    
writelog(f, 'END OF TRAINING')
f.close()
