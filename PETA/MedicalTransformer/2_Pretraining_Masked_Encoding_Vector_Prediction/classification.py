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
import wandb
import pandas as pd
from torchsummary import summary

import sys
sys.path.append('../../src')
from transforms import ToLabelOutput, TransformGridImage, TransformReduced3DImage, Transform3DImage, MinMaxNormalization
from datasets import ADNIDataset
from dl_builder import DLBuilder
from datasets_pretrain import PretrainDatasetBuilder
from cross_validation import getKFoldTrainAndValDatasets
from classification_lib import Multiview_Classification

SETS_FOLDER = "../../Sets"
IMAGES_ROOT_FOLDER = "../../.."

def getCSVs(args):
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

    return truthLabel, trainDatasetCSV, valDatasetCSV, testDatasetCSV

def getNormalization(args):
    if args.normalization == "min-max":
        normalization = MinMaxNormalization(args.min_intensity, 1)
        writelog(f, f"Normalization: MinMax({args.min_intensity}, 1)")
    elif args.normalization == "z-score":
        trainMean = 0.26102542877197266
        trainStd = 0.46347398656747363
        normalization = transforms.Normalize([trainMean], [trainStd])
        writelog(f, f"Normalization: Z-Score({mean}, {std})")
    else:
        raise Exception(f"Unknown normalization {args.normalization}")
    return normalization

def getDataAugmentation(args):
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
    return dataAugmentation

def get3DTransforms(args):
    if args.axial_slicelen == 16:
        trainTransform3D = TransformReduced3DImage(**dataAugmentation)
        valTransform3D = TransformReduced3DImage()
    else:
        # En el caso de Fleni600, tenemos que setear el resample Z en el caso de que esté seteado
        resampleZ = None
        if args.dataset == "fleni600":
            resampleZ = args.fleni_resamplez
        else:
            resampleZ = args.adni_resamplez
        trainTransform3D = Transform3DImage(yDim = args.axial_slicelen, **dataAugmentation, augmentation = args.augmentation, resampleZ = resampleZ)
        valTransform3D = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no', resampleZ = resampleZ)
    return trainTransform3D, valTransform3D

def getTrainLoader(args, trainDatasetCSV):
    if args.dataset == 'adni':
        train_csv = pd.read_csv(f"../../Sets/{trainDatasetCSV}")

        if args.miniset_size != None:
            train_csv = train_csv.sample(n=args.miniset_size, random_state = random_state)
            print(f"Reducing adni set to fixed n = {args.miniset_size} and random_state = {random_state}")
            print("Selected samples:")
            print(str(train_csv['Image Data ID']))

        train_dataset = ADNIDataset('train', train_csv, imagesDir, transform = trainTransform, target_transform = ToLabelOutput(numClasses = 2), truthLabel = truthLabel)
        trainLoader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True, 
                                pin_memory=True,
                                drop_last=drop_last)

        writelog(f, f"Train samples: {len(train_dataset)}")
        writelog(f, str(pd.read_csv(f"../../Sets/{trainDatasetCSV}")))
    elif args.dataset == 'adni-plus':
        print("with replacement")
        print(args.with_replacement)
        pretrainConfiguration = {
            "adniMini": False,
            "adni": True,
            "fleni": False,
            "merida": True,
            "chinese": True,
            "subset_size": args.subset_size, 
            "num_classes": 2,
            "axial_slicelen": args.axial_slicelen,
            "miniset_size": args.miniset_size,
            "replacement": args.with_replacement,
            "adniTruthLabel": truthLabel,
        }
        dataLoaderArgs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "shuffle": False, # falso porque usa PretrainDatasetBuilder
            "pin_memory": True,
            "drop_last": drop_last
        }
        writelog(f, "Pretrain configuration: ")
        writelog(f, str(pretrainConfiguration))
        train_dataset, trainLoader = PretrainDatasetBuilder(dataAugmentation, dataLoaderArgs, normalization, augmentation = args.augmentation, random_state = random_state, adniDatasetCSV = f"../../Sets/{trainDatasetCSV}", **pretrainConfiguration).build()

        writelog(f, f"Train samples: {len(train_dataset)}")
        writelog(f, str(pd.read_csv(f"../../Sets/{trainDatasetCSV}")))
    elif args.dataset == 'fleni600':
        dataLoaderArgs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "shuffle": True, 
            "pin_memory": True,
            "drop_last": drop_last
        }
        dl_builder = DLBuilder(SETS_FOLDER, IMAGES_ROOT_FOLDER, dataLoaderArgs, normalization, gridTransform = trainTransform3D, verbose = False)
        trainLoader = dl_builder.fleni600_train()
    else:
        raise Exception(f"No known dataset {args.dataset}")
    return trainLoader


def getEvalLoader(args, valDatasetCSV):
    if args.eval_dataset == 'adni':
       writelog(f, "Using full ADNI dataset as eval")
       val_csv = f"../../Sets/{valDatasetCSV}"
    elif args.eval_dataset == 'adni_train':
       # Por si queremos testear consistencia
        writelog(f, "Using full ADNI TRAIN dataset as eval")
        val_csv = '../../Sets/Muestra3700_80_10_10_dxlast_train.csv'
        val_csv = pd.read_csv(val_csv)
        if args.miniset_size != None:
            val_csv = val_csv.sample(n=args.miniset_size, random_state = random_state)
            writelog(f, f"Reducing adni VAL set to fixed n = {args.miniset_size} and random_state = {random_state}")
            writelog(f, "Selected samples:")
            writelog(f, str(val_csv['Image Data ID']))
    elif args.eval_dataset == 'mini':
       writelog(f, "Using mini eval dataset")
       val_csv = '../../Sets/Muestra3700_80_10_10_dxlast_val_mini.csv'
    elif args.eval_dataset == 'mini2':
       writelog(f, "Using mini eval dataset")
       val_csv = '../../Sets/Muestra3700_80_10_10_dxlast_val_mini2.csv'
    elif args.eval_dataset == "fleni600":
       dataLoaderArgs = {
           "batch_size": args.batch_size,
           "num_workers": args.num_workers,
           "shuffle": False, 
           "pin_memory": True,
           "drop_last": drop_last
       }
       dl_builder = DLBuilder(SETS_FOLDER, IMAGES_ROOT_FOLDER, dataLoaderArgs, normalization, gridTransform = valTransform3D, verbose = False)
       evalLoader = dl_builder.fleni600_val()
    else:
        raise Exception(f"eval dataset desconocido: {args.eval_dataset}")

    if args.eval_set_size != None:
        writelog(f, f"eval_set_size != 0. Takin samples of n = {args.eval_set_size} for val")
        val_csv = pd.read_csv(val_csv)
        val_csv = val_csv.sample(n=args.eval_set_size, random_state = random_state)

    if args.eval_dataset in ["adni", "adni_train", "mini", "mini2"]:
        valid_dataset = ADNIDataset('valid', val_csv, imagesDir, transform = valTransform, target_transform = ToLabelOutput(numClasses = 2), truthLabel = truthLabel)
        
        evalLoader = DataLoader(valid_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=args.eval_shuffle,
                                pin_memory=True,
                                drop_last=drop_last)

        writelog(f, f"Validation samples: {len(valid_dataset)}")
        if isinstance(val_csv,str):
            writelog(f, str(pd.read_csv(val_csv)))
        else:
            writelog(f, str(val_csv))

    return evalLoader

def getFleni3DTransform(args):
    if args.axial_slicelen == 16:
        fleniTransform3D = TransformReduced3DImage()
    else:
        fleniTransform3D = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no', resampleZ = args.fleni_resamplez)
    return fleniTransform3D

if __name__ == '__main__':
    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--pre_dataset", type=str, default='ixi_camcan_abide')
    parser.add_argument("--model", type=str, default='7_Multiview_MEP_CN_ResNet_freeze')
    
    parser.add_argument("--is_pool", type=int, default=1)
    parser.add_argument("--normalization", type=str, default="z-score")

    parser.add_argument("--is_finetune_resnet", type=int, default=1)

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)  # 5e-4
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--lambda1", type=float, default=0.0001)
    parser.add_argument("--lambda2", type=float, default=0.0000)
    parser.add_argument("--augmentation", type=str, default='one')

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

    parser.add_argument("--pretrain-weights", type=str)

    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--save-all-epochs", action=argparse.BooleanOptionalAction)

    parser.add_argument("--eval-fleni", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-fleni100", action=argparse.BooleanOptionalAction)

    parser.add_argument("--eval-adni-test", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-fleni600-test", action=argparse.BooleanOptionalAction)

    # Canary means we are doing a canary test. Only training with a subset
    parser.add_argument("--canary", action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--no-save", action=argparse.BooleanOptionalAction)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--miniset_size", type=int, default=None) # reduzco el set de entrenamiento
    parser.add_argument("--steplr-step-size", type=int, default=1)
    parser.add_argument("--steplr-gamma", type=float, default=0.99)
    parser.add_argument("--steplr-last-epoch", type=int, default=-1)
    parser.add_argument("--eval-set-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-shuffle", type=int, default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--min-intensity", type=float, default = -1.0)
    parser.add_argument("--dataset", type=str, default='adni')
    parser.add_argument("--eval-dataset", type=str, default='adni')
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--fc-layers", type=int, default=2)
    parser.add_argument("--truth-label", type=str, default="last")

    parser.add_argument("--collapse-ydim", type=float, default = 0.0)

    parser.add_argument("--with-replacement", action=argparse.BooleanOptionalAction, default = False)

    parser.add_argument("--subset_size", type=int, default=None) # cuantos elementos por epoch

    parser.add_argument("--adni-resamplez", type=float, default = None)
    parser.add_argument("--fleni-resamplez", type=float, default=0.610389)

    parser.add_argument("--da-yiming", action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--summary", action=argparse.BooleanOptionalAction)

    # Cross validation
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cross-validation-k", type=int, default=1)
    
    args = parser.parse_args()

    # GPU Configuration
    # gpu_id = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Tensor Seed
    if args.seed > 0:
        print(f"Using seed = {args.seed}")
        random_state = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        writelog(f, "No seed selected")
        random_state = None

    # Logging purpose
    date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
    if args.name:
        experimentName = args.name
        if args.cross_validation_k != 1:
            experimentName = f"{experimentName}_kFold{args.fold}"
        directory = f"./log/{experimentName}/"
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
    writelog(f, 'Lambda2: %.5f' % args.lambda2)
    writelog(f, '----------------------')
    # writelog(f, 'Fold: %d' % args.fold)
    writelog(f, 'Learning Rate: %.5f' % args.lr)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Epoch: %d' % args.epoch)
    writelog(f, '======================')
    f.close()

    if args.wandb and not args.summary:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="medicaltransformer-clf",
            entity="hugomassaroli",
            name = args.name,
            # track hyperparameters and run metadata
            config={
                "step": "classifier",
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
                "miniset_size": args.miniset_size,
                "pretrain_weights": args.pretrain_weights, 
                "eval_set_size": args.eval_set_size,
                "eval_shuffle": args.eval_shuffle,
                "class_scenario": args.class_scenario,
                "truth_label": args.truth_label,
                "dataset": args.dataset,
                "with_replacement": args.with_replacement,
                "subset_size": args.subset_size,
            }
        )

    f = open(directory + 'log.log', 'a')
    # Tensorboard Logging
    tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog//train_')
    tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog//valid_')

    # Define Loaders
    writelog(f, 'Load training data')
    writelog(f, 'Load validation data')


    model = nn.DataParallel(Multiview_Classification(args)).to(device)

    if args.summary:
        # Devuelve un summary del modelo, y sale
        summary(model, (128, args.axial_slicelen, 128))
        sys.exit(0)

    if args.pretrain_weights:
        pretrain_directory = args.pretrain_weights

        writelog(f, 'Loading pretrained weights from %s' % pretrain_directory)
        pretrained_dict = torch.load(pretrain_directory)
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict:
                writelog(f, k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        writelog(f, "-------------")
    else:
        writelog(f, "No pretrain weights. Training from scratch")

    imagesDir = "/home/eiarussi/Proyectos/Fleni/ADNI-MUESTRA-FULL-stripped-preprocessed3/"

    truthLabel, trainDatasetCSV, valDatasetCSV, testDatasetCSV = getCSVs(args)
    
    writelog(f, f"Truth label: {args.truth_label}")

    normalization = getNormalization(args)

    # test train
    dataAugmentation = getDataAugmentation(args)

    writelog(f, f"Data augmentation strategy: {args.augmentation}")
    writelog(f, f"Data augmentation params: {str(dataAugmentation)}")

    trainTransform3D, valTransform3D = get3DTransforms(args)
        
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

    num_classes = 2
    drop_last = False
    if args.cross_validation_k != 1:
        if args.dataset == 'fleni600':
            raise Exception("CrossValidation no está implementado con Fleni600")
        if args.dataset == 'adni-plus':
            raise Exception("CrossValidation no está implementado con Fleni600")
        
        # Cross validation
        writelog(f, f"Cross validation: {args.fold}/{args.cross_validation_k}")

        train_sets, val_sets = getKFoldTrainAndValDatasets(SETS_FOLDER + "/" + trainDatasetCSV, SETS_FOLDER + "/" + valDatasetCSV, K = args.cross_validation_k)
        trainDatasetCSV = train_sets[args.fold]
        valDatasetCSV = val_sets[args.fold]

        print(f"Len de train dataset: {len(trainDatasetCSV)}")
        print(f"Len de val dataset: {len(valDatasetCSV)}")
        print(trainDatasetCSV)
        print(valDatasetCSV)

        # TODO: esto NO funcionará para Fleni600, falta hacer esto para Fleni600
        ads = len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'AD'])
        mci = len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'MCI']) + len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'LMCI']) + len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'EMCI'])
        cns = len(trainDatasetCSV[trainDatasetCSV[truthLabel] == 'CN'])

        weights = [cns, ads, mci]  # Órden: CN, AD, MCI
        print(f"Cross entrophy weights: {weights}")

        trainDataset = ADNIDataset('trainDL', trainDatasetCSV, imagesDir, transform = trainTransform, target_transform =ToLabelOutput(num_classes), cacheSize = 0, truthLabel = truthLabel)
        evalDataset = ADNIDataset('valDL', valDatasetCSV, imagesDir, transform = valTransform, target_transform =ToLabelOutput(num_classes), cacheSize = 0, truthLabel = truthLabel )

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        evalLoader = torch.utils.data.DataLoader(evalDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        # Fin de cross validation
    else:
        trainLoader = getTrainLoader(args, trainDatasetCSV)
        evalLoader = getEvalLoader(args, valDatasetCSV)
    
    
    # Other data loaders
    phases = ['train', 'val']
    phaseDL = {
        'train': trainLoader,
        'val': evalLoader,
    }

    dlArgs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": drop_last
    }

    writelog(f, f"ADNI resampleZ = {args.adni_resamplez}")
    writelog(f, f"Fleni resampleZ = {args.fleni_resamplez}")
    fleniTransform3D = getFleni3DTransform(args)
    dlFleniBuilder = DLBuilder(SETS_FOLDER, IMAGES_ROOT_FOLDER, dlArgs, normalization, gridTransform = fleniTransform3D)

    if args.eval_fleni and args.eval_fleni100:
        raise Exception(f"Not possible to use eval_fleni and eval_fleni100 at the same time")

    if args.eval_fleni:
        phaseDL['fleni100'] = dlFleniBuilder.fleni100() 
        phaseDL['fleni60'] = dlFleniBuilder.fleni60()
        phases.append('fleni100')
        phases.append('fleni60')

    if args.eval_fleni100:
         phaseDL['fleni100'] = dlFleniBuilder.fleni100()
         phases.append('fleni100')

    if args.eval_fleni600_test:
         phaseDL['fleni600_test'] = dlFleniBuilder.fleni600_test()
         phases.append('fleni600_test')

    dlBuilder = DLBuilder(SETS_FOLDER, IMAGES_ROOT_FOLDER, dlArgs, normalization, gridTransform = valTransform3D)
        
    if args.eval_adni_test:
        adniTest = dlBuilder.adni('test', normalization, valDatasetCSV, testDatasetCSV, truthLabel)
        phaseDL['adni-test'] = adniTest
        phases.append('adni-test')

    writelog(f, "Phases:")
    writelog(f, str(phases))

    last_phase = phases[len(phases) - 1]

    if args.dataset == 'plus':
        # 116 chinese
        weights[0] += 116
        # 37 merida
        weights[0] += 37
    
    if num_classes == 2 and len(weights) == 3:
        weights = [weights[0] + weights[2], weights[1]]

    writelog(f, f"Cross entrophy weights: {weights}")
    criterion = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float32).cuda())

    # We want to optimize everything
    writelog(f, "Params to optimize: ")
    params_to_update = model.parameters()
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            writelog(f, f"\t{name}")

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    since = time.time()
    writelog(f, "Train")

    if args.save_all_epochs:
        writelog(f, f"Saving all epochs")

    # Loop over the minibatch\
    best_selection_criteria = 0.0
    best_selection_performance = None
    ES = EarlyStopping(delta=0, patience=args.patience, verbose=True)
    for epoch in range(args.epoch):
        start_epoch_time = time.time()
        writelog(f, f"Epoch {epoch}/{args.epoch}")

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            loader = phaseDL[phase]

            running_loss = 0.0
            running_corrects = 0
            labels_aggregate = torch.empty((0), dtype=torch.int)
            outputs_aggregate = torch.empty((0,2), dtype=torch.float)

            for i, xbatch in enumerate(tqdm(loader)):
                # print(xbatch)
                inputs = xbatch[0].float().cuda()
                # Needs to be parsed to long for some reason
                labels = xbatch[1].long().cuda()

                # print(labels)

                # labels = torch.tensor(a, dtype=torch.long).cuda()

                optimizer.zero_grad()

                outputs = model(inputs)

                labels_aggregate = torch.cat((labels_aggregate, labels.cpu().detach()), 0)
                outputs_aggregate = torch.cat((outputs_aggregate, outputs.cpu().detach()), 0)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # print(preds)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += + torch.sum(preds == labels.data)

                # fileName = os.path.join(experimentOutputFolder, experimentExecutionName + '_epoch' + str(epoch) + '.pth')
                # torch.save(model.state_dict(), fileName)

                # print(res)
            # print(running_loss)
            # print(len(loader.dataset))
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
            
            selection_criteria_value = auc

            weightsFile = directory + '/model/%s_%d.pt' % (args.model, epoch)
            
            if phase == 'train' and args.save_all_epochs:
                writelog(f, "Saving model (save all): " +  weightsFile)
                torch.save(model.state_dict(), weightsFile)

            if phase == 'val':
                ES(None, selection_criteria_value)

            # Writing performance no matter what is the epoch
            writelog(f, '{} Loss: {:.4f} Acc: {:.4f} BAcc: {:4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_acc, balacc, auc))

            if phase == 'val' and selection_criteria_value > best_selection_criteria:
                best_selection_criteria = selection_criteria_value
                best_selection_criteria_epoch = epoch
                best_selection_performance = performance
                if (not args.save_all_epochs) and (not args.no_save):
                    writelog(f, "Saving model: " +  weightsFile)
                    torch.save(model.state_dict(), weightsFile)
                   
                writelog(f, 'Best selection criteria is found! Validation loss : %f Epoch criteria: %f' % (epoch_loss, selection_criteria_value))
                writelog(f, "AUC: {0}\nAUPRC: {1}\nAcc: {2}\nBal Acc: {3}\nSensibility: {4}\nSpecificity: {5}\nPrecision: {6}\nRecall: {7}\nF1: {8}".format(auc, auprc, acc, balacc, sen, spec, prec, recall, f1))
                writelog(f, "Confusion matrix:\n{0}".format(cmatrix))
                
                if (not args.save_all_epochs) and (not args.no_save):
                    writelog(f, 'Models at Epoch %d are saved!' % epoch)

            if args.wandb:
                logObj = {f"{phase}_loss": epoch_loss, "best_selection_criteria": best_selection_criteria, "epoch": epoch, f"{phase}_auc": auc, f"{phase}_aucprc": auprc, f"{phase}_acc": acc, f"{phase}_balacc": balacc, f"{phase}_sen": sen, f"{phase}_spec": spec, f"{phase}_recall": recall, f"{phase}_f1": f1 }
                if phase == last_phase:
                    commit = True
                else:
                    commit = False

                wandb.log(logObj, commit = commit)

        total_epoch_time = time.time() - start_epoch_time
        writelog(f, "Epoch time: %.2f m" % (total_epoch_time / 60))
        writelog(f, "Remaining expected time: %.2f hs" % (total_epoch_time * (args.epoch - epoch)/ 60 / 60))
                
        if ES.early_stop == True:
            break
                
        writelog(f, '---')
        
    time_elapsed = time.time() - since
    avg_time_per_epoch = time_elapsed / args.epoch
    writelog(f, 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    writelog(f, 'Time per epoch {:.0f}m {:.0f}s'.format(avg_time_per_epoch // 60, time_elapsed % 60))
    writelog(f, 'Best val {}: {:4f} on epoch {}'.format("auc", best_selection_criteria, best_selection_criteria_epoch))

    if args.wandb:
        wandb.log({"time_per_epoch": avg_time_per_epoch, "total_train_time": time_elapsed})
        wandb.finish()

    writelog(f, 'END OF TRAINING')
    f.close()
