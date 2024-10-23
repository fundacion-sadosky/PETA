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

import sys
sys.path.append('../../src')
from dl_builder import DLBuilder
from transforms import ToLabelOutput, TransformGridImage, TransformReduced3DImage, Transform3DImage, MinMaxNormalization, ToLabelOutputFleni, ToLabelOutputConfigurable
from datasets import FleniMyriamDataset, BaseDataset, ADNIDataset
from classification import Multiview_Classification

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--pre_dataset", type=str, default='ixi_camcan_abide')
parser.add_argument("--model", type=str, default='7_Multiview_MEP_CN_ResNet_freeze')
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--is_pool", type=int, default=1)

parser.add_argument("--is_finetune_resnet", type=int, default=1)

parser.add_argument("--mask_ratio", type=float, default=0.1)
# parser.add_argument("--sample_ratio", type=float, default=0.5)

parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=5e-4)  # 5e-4
parser.add_argument("--batch_size", type=int, default=64)
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
parser.add_argument("--weights", type=str)
parser.add_argument("--normalization", type=str, default="z-score")
parser.add_argument("--truth-label", type=str, default="last")
parser.add_argument("--min-intensity", type=float, default = -1.0)
parser.add_argument("--fc-layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--val", action=argparse.BooleanOptionalAction)
parser.add_argument("--test", action=argparse.BooleanOptionalAction)
parser.add_argument("--fleni60", action=argparse.BooleanOptionalAction)
parser.add_argument("--fleni600", action=argparse.BooleanOptionalAction)
parser.add_argument("--fleni600-test", action=argparse.BooleanOptionalAction)
parser.add_argument("--fleni100", action=argparse.BooleanOptionalAction)
parser.add_argument("--adni-resamplez", type=float, default=None)
parser.add_argument("--fleni-resamplez", type=float, default=0.610389)

args = parser.parse_args()

# GPU Configuration
# gpu_id = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = nn.DataParallel(Multiview_Classification(args)).to(device)

weights = args.weights

print(f"Loading weights from {weights}")

num_classes = 2
validationCacheSize = 0

weights_dict = torch.load(weights)
model_dict = model.state_dict()
for k, v in weights_dict.items():
    if k in model_dict:
        print(k)
model.load_state_dict(weights_dict)

print(f"ADNI resample Z: {args.adni_resamplez}")
print(f"Fleni resample Z: {args.fleni_resamplez}")

fleni100ImagesFolder = "/home/eiarussi/Proyectos/Fleni/fleni-stripped-preprocessed3/"
fleni60ImagesFolder = "/home/eiarussi/Proyectos/Fleni/fleni-stripped-preprocessed4/"

valMyryamDatasetCSV = "fleni-myriam-curated.csv"
valFleni60DatasetCSV = "fleni-PET_clasificados60.csv"

# ADNI val

if args.normalization == "min-max":
    normalization = MinMaxNormalization(args.min_intensity, 1)
    print(f"Normalization: MinMax({args.min_intensity}, 1)")
elif args.normalization == "z-score":
    trainMean = 0.26102542877197266
    trainStd = 0.46347398656747363
    normalization = transforms.Normalize([trainMean], [trainStd])
    print(f"Normalization: Z-Score({trainMean}, {trainStd})")
else:
    raise Exception(f"Unknown normalization {args.normalization}")


if args.axial_slicelen == 16:
    adniTransform = TransformReduced3DImage()
else:
    adniTransform = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no', resampleZ = args.adni_resamplez)
    
valTransform = torchvision.transforms.Compose([
    adniTransform,
    torchvision.transforms.ToTensor(),
    normalization
])

imagesDir = "/home/eiarussi/Proyectos/Fleni/ADNI-MUESTRA-FULL-stripped-preprocessed3/"

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
print(f"Truth label: {args.truth_label}")

adni_val_dataset = ADNIDataset('valid', f"../../Sets/{valDatasetCSV}", imagesDir, transform = valTransform, target_transform = ToLabelOutput(numClasses = 2), truthLabel = truthLabel) 
adniValLoader = DataLoader(adni_val_dataset,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# ADNI test

if args.normalization == "min-max":
    normalization = MinMaxNormalization(args.min_intensity, 1)
    print(f"Normalization: MinMax({args.min_intensity}, 1)")
elif args.normalization == "z-score":
    trainMean = 0.26102542877197266
    trainStd = 0.46347398656747363
    normalization = transforms.Normalize([trainMean], [trainStd])
    print(f"Normalization: Z-Score({trainMean}, {trainStd})")
else:
    raise Exception(f"Unknown normalization {args.normalization}")

testTransform = torchvision.transforms.Compose([
    adniTransform,
    torchvision.transforms.ToTensor(),
    normalization
])


adni_test_dataset = ADNIDataset('testid', f"../../Sets/{testDatasetCSV}", imagesDir, transform = testTransform, target_transform = ToLabelOutput(numClasses = 2), truthLabel = truthLabel) 
adniTestLoader = DataLoader(adni_test_dataset,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

if args.axial_slicelen == 16:
    fleniTransform = TransformReduced3DImage()
else:
    fleniTransform = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no', resampleZ = args.fleni_resamplez)

# Fleni 100

if args.normalization == "min-max":
    normalizationFleni100 = MinMaxNormalization(args.min_intensity, 1)
    print(f"Normalization Fleni100: MinMax({args.min_intensity}, 1)")
elif args.normalization == "z-score":
    meanFleni100 = 3383.637939453125
    stdFleni100 = 7348.981153874324
    normalizationFleni100 = transforms.Normalize([meanFleni100], [stdFleni100])
    print(f"Normalization Fleni100: Z-Score({meanFleni100}, {stdFleni100})")
else:
    raise Exception(f"Unknown normalization {args.normalization}")

valTransformFleni100 = torchvision.transforms.Compose([
    fleniTransform,
    torchvision.transforms.ToTensor(),
    normalizationFleni100
])

fleni100_dataset = FleniMyriamDataset('fleni100', f"../../Sets/{valMyryamDatasetCSV}", fleni100ImagesFolder, transform = valTransformFleni100, target_transform =ToLabelOutputFleni(num_classes), cacheSize = validationCacheSize )
fleni100Loader = DataLoader(fleni100_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

# Fleni 60

if args.normalization == "min-max":
    normalizationFleni60 = MinMaxNormalization(args.min_intensity, 1)
    print(f"Normalization Fleni60: MinMax({args.min_intensity}, 1)")
elif args.normalization == "z-score":
    meanFleni60 = 3383.637939453125
    stdFleni60 = 7348.981153874324
    normalizationFleni60 = transforms.Normalize([meanFleni60], [stdFleni60])
    print(f"Normalization Fleni60: Z-Score({meanFleni60}, {stdFleni60})")
else:
    raise Exception(f"Unknown normalization {args.normalization}")

valTransformFleni60 = torchvision.transforms.Compose([
    fleniTransform,
    torchvision.transforms.ToTensor(),
    normalizationFleni60
])

dicti = {
    "AD": 1,
    "non-AD": 0
}
fleni60_dataset = BaseDataset('fleni60', f"../../Sets/{valFleni60DatasetCSV}", fleni60ImagesFolder, studyIDLabel = 'anon_id', transform = valTransformFleni60, target_transform = ToLabelOutputConfigurable(dicti), truthLabel = 'Conclusion PET')
fleni60Loader = DataLoader(fleni60_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

dlArgs = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "shuffle": False,
    "pin_memory": True,
    "drop_last": False
}
SETS_FOLDER = "../../Sets"
IMAGES_ROOT_FOLDER = "../../.."
dl_builder = DLBuilder(SETS_FOLDER, IMAGES_ROOT_FOLDER, dlArgs, normalization = normalization, gridTransform = fleniTransform)
fleni600Loader = dl_builder.fleni600()

fleni600TestLoader = dl_builder.fleni600_test()

criterion = nn.CrossEntropyLoss(torch.tensor([765.0 + 1229.0, 1039.0]).cuda())

def eval(name, loader):
    print(f"Evaluating {name}")
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    labels_aggregate = torch.empty((0), dtype=torch.int)
    outputs_aggregate = torch.empty((0,2), dtype=torch.float)

    with torch.no_grad():
        for i, xbatch in enumerate(tqdm(loader)):
            inputs = xbatch[0].float().cuda()
            labels = xbatch[1].cuda()

            outputs = model(inputs)

            labels_aggregate = torch.cat((labels_aggregate, labels.cpu().detach()), 0)
            outputs_aggregate = torch.cat((outputs_aggregate, outputs.cpu().detach()), 0)

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += + torch.sum(preds == labels.data)  

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)
    
    performance = calculate_performance(labels_aggregate.cpu().detach().numpy(), outputs_aggregate.cpu().detach().numpy(), args)[args.class_scenario]

    auc = performance[0]
    auprc = performance[1]
    acc = performance[2]
    balacc = performance[3]
    sen = performance[4]
    spec = performance[5]
    prec = performance[6]
    recall = performance[7]
    f1 = performance[8]
    cmatrix = performance[9] 
    print("AUC: {0}\nAUPRC: {1}\nAcc: {2}\nBal Acc: {3}\nSensibility: {4}\nSpecificity: {5}\nPrecision: {6}\nRecall: {7}\nF1: {8}".format(auc, auprc, acc, balacc, sen, spec, prec, recall, f1))
    print("Confusion matrix:\n{0}".format(cmatrix))
    print("--------------")

# eval("ADNI Train", adniTrainLoader)

if args.val:
    eval("ADNI Val", adniValLoader)

if args.test:
    eval("ADNI Test", adniTestLoader)

if args.fleni60:
    eval("Fleni60", fleni60Loader)

if args.fleni100:
    eval("Fleni100", fleni100Loader)

if args.fleni600:
    eval("Fleni600", fleni600Loader)

if args.fleni600_test:
    eval("Fleni600Test", fleni600TestLoader)
