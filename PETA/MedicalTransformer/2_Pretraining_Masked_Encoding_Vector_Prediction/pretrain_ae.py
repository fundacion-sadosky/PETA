import torch.optim as optim
from models.ResNet_Model import Multiview_MEP
from losses import *
from helpers import *
import os
import random
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tqdm import tqdm
import warnings
import torchvision
from torchvision import transforms
warnings.filterwarnings("ignore")

import sys
sys.path.append('../../src')
from transforms import ToLabelOutput, TransformGridImage, TransformReduced3DImage, Transform3DImage, MinMaxNormalization
from datasets import ADNIDataset
from datasets_pretrain import PretrainDatasetBuilder
from dl_builder import DLBuilder
import pandas as pd
import wandb

SETS_FOLDER = "../../Sets"
IMAGES_ROOT_FOLDER = "../../.."

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--pre_dataset", type=str, default='ixi_camcan_abide')
parser.add_argument("--model", type=str, default='7_Multiview_MEP_CN_ResNet_freeze')
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--is_pool", type=int, default=1)
parser.add_argument("--normalization", type=str, default="z-score")

parser.add_argument("--is_finetune_resnet", type=int, default=1)

parser.add_argument("--mask_ratio", type=float, default=0.1)
# parser.add_argument("--sample_ratio", type=float, default=0.5)

parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-4)  # 5e-4
parser.add_argument("--batch_size", type=int, default=4)
# parser.add_argument("--lambda1", type=float, default=0.0001)
parser.add_argument("--lambda2", type=float, default=0.0000)

parser.add_argument("--depth", type=int, default=18)
parser.add_argument("--inplanes", type=int, default=16)
parser.add_argument("--d_f", type=int, default=64)

# Transformer
# parser.add_argument("--max_slicelen", type=int, default=229)
# parser.add_argument("--axial_slicelen", type=int, default=193)
# parser.add_argument("--coronal_slicelen", type=int, default=229)

parser.add_argument("--max_slicelen", type=int, default=128)
parser.add_argument("--axial_slicelen", type=int, default=16)
parser.add_argument("--coronal_slicelen", type=int, default=128)


parser.add_argument("--d_ff", type=int, default=128)
parser.add_argument("--num_stack", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=4)
# parser.add_argument("--slice_len", type=int, default=193)

# parser.add_argument("--class_scenario", type=str, default='cn_mci_ad')
# parser.add_argument("--class_scenario", type=str, default='mci_ad')
# parser.add_argument("--class_scenario", type=str, default='cn_mci')
parser.add_argument("--class_scenario", type=str, default='cn_ad')

parser.add_argument("--prepretrain-weights", type=str)

parser.add_argument("--dataset", type=str, default='mixed')

parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)

parser.add_argument("--eval-dataset", type=str, default='adni')
parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
# 2*this is the number of samples every epoch
# if None, the whole dataset is used
parser.add_argument("--subset_size", type=int, default=None) # cuantos elementos por epoch
parser.add_argument("--miniset_size", type=int, default=None) # reduzco el set de entrenamiento
parser.add_argument("--save-best", type=int, default=1)
parser.add_argument("--save-every", type=int, default=-1)

parser.add_argument("--augmentation", type=str, default='all')
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--steplr-step-size", type=int, default=1)
parser.add_argument("--steplr-gamma", type=float, default=0.99)
parser.add_argument("--steplr-last-epoch", type=int, default=-1)

parser.add_argument("--eval-set-size", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--eval-shuffle", type=int, default=1)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--min-intensity", type=float, default = -1.0)
parser.add_argument("--patience", type=int, default = 30)
parser.add_argument("--truth-label", type=str, default="last")
parser.add_argument("--collapse-ydim", type=float, default = 0.0)
parser.add_argument("--start-from", type=int, default = 0)
parser.add_argument("--with-replacement",  action=argparse.BooleanOptionalAction)
parser.add_argument("--eval-fleni", action=argparse.BooleanOptionalAction)
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
parser.add_argument("--da-yiming", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# GPU Configuration
# gpu_id = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Load Dataset
# images = np.load('/home/eiarussi/Proyectos/Fleni/MedicalTransformer/images.npy', mmap_mode="r", allow_pickle=True)  # (566, 193, 229, 193)
# indexes = np.load('/home/eiarussi/Proyectos/Fleni/MedicalTransformer/indices.npy', allow_pickle=True)  # (509,), (57,)
labels = []

# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))

if args.name:
    directory = './log/%s/' % args.name
else:
    directory = './log/%s/%s_batch_%d_lr_%f_lambda2_%f_ResNet_%d_inplanes_%d/' % (args.pre_dataset, date_str, args.batch_size, args.lr, args.lambda2, args.depth, args.inplanes)

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
writelog(f, 'Model: %s' % (args.model))
writelog(f, 'Log directory: %s' % directory)
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, 'Is finetune resnet: %d' % args.is_finetune_resnet)
writelog(f, '----------------------')
# writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '======================')
f.close()

if args.wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="medicaltransformer-tx",
        entity="hugomassaroli",
        name = args.name,
        # track hyperparameters and run metadata
        config={
            "step": "transformers",
            "learning_rate": args.lr,
            "seed": args.seed,
            "augmentation": args.augmentation,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "lambda2": args.lambda2,
            "depth": args.depth,
            "inplanes": args.inplanes,
            "d_f": args.d_f,
            "max_slicelen": args.max_slicelen,
            "axial_slicelen": args.axial_slicelen,
            "coronal_slicelen": args.coronal_slicelen,
            "dataset": args.dataset,
            "eval_dataset": args.eval_dataset,
            "subset_size": args.subset_size,
            "miniset_size": args.miniset_size,
            "pretrain_weights": args.prepretrain_weights, 
            "eval_set_size": args.eval_set_size,
            "eval_shuffle": args.eval_shuffle,
            "patience": args.patience,
            "truth_label": args.truth_label,
        }
    )

f = open(directory + 'log.log', 'a')
# Tensorboard Logging
# tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/train_')
# tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/valid_')
# tfw_test = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/test_')
tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog//train_')
tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog//valid_')

# Tensor Seed
if args.seed > 0:
    writelog(f, f"Using seed = {args.seed}")
    random_state = args.seed
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
    trainTransform3D = TransformReduced3DImage(**dataAugmentation)
    valTransform3D = TransformReduced3DImage()
else:
    trainTransform3D = Transform3DImage(yDim = args.axial_slicelen, **dataAugmentation, augmentation = args.augmentation)
    valTransform3D = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no')
    
# Define Loaders
trainTransform = torchvision.transforms.Compose([
    trainTransform3D,
    torchvision.transforms.ToTensor(),
    normalization
])

valTransform = torchvision.transforms.Compose([
    valTransform3D,
    torchvision.transforms.ToTensor(),
    normalization
])

imagesDir = "/home/eiarussi/Proyectos/Fleni/ADNI-MUESTRA-FULL-stripped-preprocessed3/"

writelog(f, 'Load training data')

drop_last = False
dataLoaderArgs = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "shuffle": False, # falso porque usa PretrainDatasetBuilder
    "pin_memory": True,
    "drop_last": drop_last
}

pretrainADNI = False
pretrainFleni = False
pretrainADNIMini = False
pretrainMerida = False
pretrainChinese = False
pretrainADNIMini = False

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

writelog(f, 'Load validation data')
# valid_loader = sample_loader('valid', images, labels, indexes, args, shuffle=False)

# Checking truth label
if args.truth_label == "last":
    truthLabel = "DX_last"
    trainDatasetCSV = "Muestra3700_80_10_10_dxlast_train.csv"
    valDatasetCSV = "Muestra3700_80_10_10_dxlast_val.csv"
    # testDatasetCSV = "Muestra3700_80_10_10_dxlast_test.csv"
elif args.truth_label == "visit":
    truthLabel = "DX_vis"
    trainDatasetCSV = "Muestra3700_80_10_10_dxvisit953_train.csv"
    valDatasetCSV = "Muestra3700_80_10_10_dxvisit953_val.csv"
    # testDatasetCSV = "Muestra3700_80_10_10_dxvisit953_test.csv"
else:
    raise Exception(f"Not supported truth label = {args.truth_label}")
writelog(f, f"Truth label: {args.truth_label}")

pretrainConfiguration = {
    "adniMini": pretrainADNIMini,
    "adni": pretrainADNI,
    "fleni": pretrainFleni,
    "merida": pretrainMerida,
    "chinese": pretrainChinese,
    "subset_size": args.subset_size, # el mas chico, en este caso Fleni 
    "num_classes": 2,
    "axial_slicelen": args.axial_slicelen,
    "miniset_size": args.miniset_size,
    "replacement": args.with_replacement,
    "adniTruthLabel": truthLabel,
    "verbose": args.verbose,
}
writelog(f, "Pretrain configuration: ")
writelog(f, str(pretrainConfiguration))
train_dataset, train_loader = PretrainDatasetBuilder(dataAugmentation, dataLoaderArgs, normalization, augmentation = args.augmentation, random_state = random_state, adniDatasetCSV = f"../../Sets/{trainDatasetCSV}", **pretrainConfiguration).build()

writelog(f, f"Train samples: {len(train_dataset)}")
writelog(f, str(pd.read_csv(f"../../Sets/{trainDatasetCSV}")))

if args.eval_dataset == 'adni':
    writelog(f, "Using full ADNI dataset as eval")
    val_csv = f"../../Sets/{valDatasetCSV}"
    writelog(f, f"Using {valDatasetCSV}")
elif args.eval_dataset == 'adni_train':
    # Por si queremos testear consistencia
    writelog(f, "Using full ADNI TRAIN dataset as eval")
    val_csv = f"../../Sets/{trainDatasetCSV}" 
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
    writelog(f, f"eval_set_size != 0. Takin samples of n = {args.eval_set_size} for val")
    val_csv = pd.read_csv(val_csv)
    val_csv = val_csv.sample(n=args.eval_set_size, random_state = random_state)

if isinstance(val_csv,str):
    writelog(f, str(pd.read_csv(val_csv)))
else:
    writelog(f, str(val_csv))

valid_dataset = ADNIDataset('valid', val_csv, imagesDir, transform = valTransform, target_transform = ToLabelOutput(), truthLabel = truthLabel)
valid_loader = DataLoader(valid_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle = args.eval_shuffle == 1,
                        pin_memory=True,
                        drop_last=drop_last)

writelog(f, f"Validation samples: {len(valid_dataset)}")

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

def train(dataloader, dir='.'):
    # Set mode as training
    model.train()

    # Define training variables
    train_loss = 0
    n_batches = 0

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):
        # x = Variable(xbatch['data'].float(), requires_grad=True).cuda()  # [4, 96, 114, 96]
        inputs = xbatch[0]
        labels = xbatch[1]
        x = Variable(inputs.float(), requires_grad=True).cuda()  # [B, 193, 229, 193]

        if args.is_finetune_resnet == 1:
            for param in model.module.encoding.parameters():
                # print(param)
                param.requires_grad = False
            for param in model.module.encoding_sag.parameters():
                # print(param)
                param.requires_grad = False
            for param in model.module.encoding_cor.parameters():
                # print(param)
                param.requires_grad = False
        else:
            for param in model.module.encoding.parameters():
                param.requires_grad = True
            for param in model.module.encoding_sag.parameters():
                # print(param)
                param.requires_grad = True
            for param in model.module.encoding_cor.parameters():
                # print(param)
                param.requires_grad = True

        # Zero Grad
        optimizer.zero_grad()

        emb = model(x)  # [1, 193, 193]

        # Calculate Loss
        loss = creterion(emb)
        loss.backward()
        optimizer.step()
        print('Training loss = (%.5f)' % loss)

        # Backpropagation & Update the weights
        train_loss += (loss.item() * x.size(0))
        n_batches += 1

    # Take average
    # train_loss = train_loss / n_batches
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


def evaluate(phase, dataloader, dir='.'):
    # Set mode as training
    model.eval()

    # Define training variables
    test_loss = 0
    n_batches = 0

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):
            # x = xbatch['data'].float().cuda()

            inputs = xbatch[0]
            labels = xbatch[1]
            x = Variable(inputs.float(), requires_grad=False).cuda()  # [B, 193, 229, 193]

            emb = model(x)  # [1, 193, 193]

            # Calculate Loss
            loss = creterion(emb)
            print('(%s) loss = (%.5f)' % (phase, loss))

            test_loss += (loss.item() * x.size(0))
            n_batches += 1

    # Take average
    # test_loss = test_loss / n_batches
    test_loss = test_loss / len(dataloader.dataset)
    writelog(f, '%s Loss: %.8f' % (phase, test_loss))

    # Tensorboard Logging
    info = {'loss': test_loss}

    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        if phase == 'Validation':
            tfw_valid.add_summary(summary, epoch)
        # else:
        #     tfw_test.add_summary(summary, epoch)

    return test_loss


# Define Model
model = nn.DataParallel(Multiview_MEP(args)).to(device)

# pretrain_directory = '/DataCommon/ejjun/MedBERT/experiment/0_prepretrain/log/ixi_camcan_abide/20210129.19.22.57_batch_8_lr_0.000100_lambda2_0.001000_ResNet_18_inplanes_16/model/0_ResNet_prepretrain.pt'
# pretrain_directory = '/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/ixi_camcan_abide/25d/20230713.14.36.49_batch_32_lr_0.000500_lambda2_0.000000_ResNet_18_inplanes_16/model/prepretrain_model.pt'
#pretrain_directory = '/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/ixi_camcan_abide/25d/20230717.12.11.53_batch_32_lr_0.000100_lambda2_0.000000_ResNet_18_inplanes_16/model/prepretrain_model.pt'
# pretrain_directory = '/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/ixi_camcan_abide/25d/20230722.10.45.41_batch_32_lr_0.000100_lambda2_0.000000_ResNet_18_inplanes_16/model/prepretrain_model.pt' # pretrain for solo ADNI, exp13
# pretrain_directory = '/home/eiarussi/Proyectos/Fleni/PET-IA/MedicalTransformer/1_Pretraining_Triplet_Loss/log/ixi_camcan_abide/25d/20230724.14.34.50_batch_32_lr_0.000100_lambda2_0.000000_ResNet_18_inplanes_16/model/prepretrain_model.pt' # pretrain for ADNI + fleni, exp14

pretrain_directory = args.prepretrain_weights

writelog(f, 'Loading encoding weights from %s' % pretrain_directory)

pretrained_dict = torch.load(pretrain_directory, map_location = device)
model_dict = model.state_dict()
for k, v in pretrained_dict.items():
    if k in model_dict:
        print(k)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


# loss function
creterion = MultiviewMSELoss(args)
# creterion = nn.L1Loss()

# optimizer
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

if args.start_from > 0:
    writelog(f, f"Starting from epoch {args.start_from}")

start_time = time.time()

# Train Epoch
ES = EarlyStopping(delta=0, patience=args.patience, verbose=True)
for epoch in range(args.start_from, args.epoch):
    start_epoch_time = time.time()
    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, "lr = %f" % scheduler.get_lr()[0])
    writelog(f, 'Training')
    loss_train = train(dataloaders['train'], dir=directory)

    writelog(f, 'Validation')
    loss_val = evaluate('Validation', dataloaders['valid'], dir=directory)

    if epoch == args.start_from:
        valid['loss'] = loss_val

    # Saving model very X epochs
    if args.save_every != -1:
        if epoch % args.save_every == 0:
            weightsFile = directory + 'model/%s_%d.pt' % (args.model, epoch)
            writelog(f, "Saving model: %s" % weightsFile)
            torch.save(model.state_dict(), weightsFile)

    # Save Model
    if loss_val < valid['loss']:
        # Guardo ambos, con model / epoch y con nombre genérico
        weightsFile = directory + 'model/%s_%d.pt' % (args.model, epoch)
        torch.save(model.state_dict(), weightsFile)
        weightsFile = directory + 'model/pretrain_model.pt'
        torch.save(model.state_dict(), weightsFile)
        
        writelog(f, 'Best validation loss is found! Validation loss : %f' % loss_val)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        writelog(f, "Saving model: %s" %  weightsFile)
        valid['loss'] = loss_val
        valid['epoch'] = epoch

    if args.eval_fleni:
        loss_fleni100 = evaluate('Fleni100', dataloaders['fleni100'], dir=directory)
        loss_fleni60 = evaluate('Fleni60', dataloaders['fleni60'], dir=directory)

        if args.wandb:
            wandb.log({"fleni100_loss": loss_fleni100, "fleni60_loss": loss_fleni60}, commit = False)
        
    ES(loss_val, None)

    total_epoch_time = time.time() - start_epoch_time
    writelog(f, "Epoch time: %.2f m" % (total_epoch_time / 60))
    writelog(f, "Remaining expected time: %.2f hs" % (total_epoch_time * (args.epoch - epoch)/ 60 / 60))

    if args.wandb:
        # Guardamos el val_loss mínimo hasta ahora
        wandb.log({"train_loss": loss_train, "val_loss": loss_val, "min_val_loss": valid['loss'], "epoch": epoch, "remaining_time": total_epoch_time * (args.epoch - epoch)/ 60 / 60 })

    # loss_test = evaluate('Test', dataloaders['test'], dir=directory)

    scheduler.step()
    
    if ES.early_stop == True:
        break

writelog(f, 'Best model for testing: epoch %d-th' % valid['epoch'])

end_time = time.time()
total_time = end_time - start_time
avg_time_per_epoch = total_time / args.epoch
writelog(f, "Total time: %.2f hs" % (total_time/60/60))
writelog(f, "Avg time per epoch: %.2f m" %  (avg_time_per_epoch/60))

# writelog(f, 'Testing')
# loss = evaluate('test', dataloaders['test'], dir=directory)

if args.wandb:
    wandb.log({"time_per_epoch": avg_time_per_epoch, "total_train_time": total_time})
    wandb.finish()

writelog(f, 'END OF TRAINING')
f.close()
