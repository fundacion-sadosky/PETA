import argparse
from pathlib import Path


def parse_opts():
    parser = argparse.ArgumentParser()

    ####################################################################################
    ####################################################################################
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pre_dataset", type=str, default='ixi_camcan_abide')
    parser.add_argument("--approach", type=str, default='25d')  # 2d, 25d, 3d
    # parser.add_argument("--model", type=str, default='0_ResNet_prepretrain')
    parser.add_argument("--fold", type=int, default=0)
    # parser.add_argument("--intp_ch", type=int, default="32")
    parser.add_argument("--is_pool", type=int, default=1)
    parser.add_argument("--isAugment", type=int, default=0)
    parser.add_argument("--augmentation", type=str, default='all')
    parser.add_argument("--sample_ratio", type=float, default=1)
    
    parser.add_argument("--normalization", type=str, default="z-score")
    
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)  # 5e-4
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lambda2", type=float, default=0.0000)

    parser.add_argument("--steplr-step-size", type=int, default=1)
    parser.add_argument("--steplr-gamma", type=float, default=0.99)
    parser.add_argument("--steplr-last-epoch", type=int, default=-1)

    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--inplanes", type=int, default=16)
    parser.add_argument("--d_f", type=int, default=64)

    parser.add_argument("--max_slicelen", type=int, default=128)
    parser.add_argument("--axial_slicelen", type=int, default=16)
    parser.add_argument("--coronal_slicelen", type=int, default=128)

    # parser.add_argument("--class_scenario", type=str, default='cn_mci_ad')
    # parser.add_argument("--class_scenario", type=str, default='mci_ad')
    # parser.add_argument("--class_scenario", type=str, default='cn_mci')
    parser.add_argument("--class_scenario", type=str, default='cn_ad')

    # Either mixed or adni
    parser.add_argument("--dataset", type=str, default='mixed')
    parser.add_argument("--eval-dataset", type=str, default='adni')
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    # 2*this is the number of samples every epoch
    # if None, the whole dataset is used
    parser.add_argument("--subset_size", type=int, default=None) # cuantos elementos por epoch
    parser.add_argument("--miniset_size", type=int, default=None) # reduzco el set de entrenamiento
    parser.add_argument("--eval_set_size", type=int, default=None) # cuantos de eval tengo
    parser.add_argument("--save-best", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=-1)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)

    # Para solo evaluar
    parser.add_argument("--weights", type=str, default= None) # if true, then it won't train, just eval
    parser.add_argument("--eval-set-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-shuffle", type=int, default=1)
    parser.add_argument("--eval-rounds", type=int, default=1)
    parser.add_argument("--nonlinear-prob", type=float, default=0.5)

    parser.add_argument("--name", type=str, default = None)

    parser.add_argument("--min-intensity", type=float, default = -1.0)
    parser.add_argument("--truth-label", type=str, default="last")

    parser.add_argument("--patience", type=int, default = 30)
    parser.add_argument("--with-replacement",  action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-fleni", action=argparse.BooleanOptionalAction)
    parser.add_argument("--collapse-ydim", type=float, default = 0.0)
    parser.add_argument("--da-yiming", action=argparse.BooleanOptionalAction)
    ####################################################################################
    ####################################################################################


    # parser.add_argument('--root_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Root directory path')
    # parser.add_argument('--video_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Directory path of videos')
    # parser.add_argument('--annotation_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Annotation file path')
    # parser.add_argument('--result_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Result directory path')
    # parser.add_argument(
    #     '--dataset',
    #     default='kinetics',
    #     type=str,
    #     help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    # parser.add_argument(
    #     '--n_classes',
    #     default=400,
    #     type=int,
    #     help=
    #     'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    # )
    # parser.add_argument('--n_pretrain_classes',
    #                     default=0,
    #                     type=int,
    #                     help=('Number of classes of pretraining task.'
    #                           'When using --pretrain_path, this must be set.'))
    # parser.add_argument('--pretrain_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Pretrained model path (.pth).')
    # parser.add_argument(
    #     '--ft_begin_module',
    #     default='',
    #     type=str,
    #     help=('Module name of beginning of fine-tuning'
    #           '(conv1, layer1, fc, denseblock1, classifier, ...).'
    #           'The default means all layers are fine-tuned.'))
    # parser.add_argument('--sample_size',
    #                     default=112,
    #                     type=int,
    #                     help='Height and width of inputs')
    # parser.add_argument('--sample_duration',
    #                     default=16,
    #                     type=int,
    #                     help='Temporal duration of inputs')
    # parser.add_argument(
    #     '--sample_t_stride',
    #     default=1,
    #     type=int,
    #     help='If larger than 1, input frames are subsampled with the stride.')
    # parser.add_argument(
    #     '--train_crop',
    #     default='random',
    #     type=str,
    #     help=('Spatial cropping method in training. '
    #           'random is uniform. '
    #           'corner is selection from 4 corners and 1 center. '
    #           '(random | corner | center)'))
    # parser.add_argument('--train_crop_min_scale',
    #                     default=0.25,
    #                     type=float,
    #                     help='Min scale for random cropping in training')
    # parser.add_argument('--train_crop_min_ratio',
    #                     default=0.75,
    #                     type=float,
    #                     help='Min aspect ratio for random cropping in training')
    # parser.add_argument('--no_hflip',
    #                     action='store_true',
    #                     help='If true holizontal flipping is not performed.')
    # parser.add_argument('--colorjitter',
    #                     action='store_true',
    #                     help='If true colorjitter is performed.')
    # parser.add_argument('--train_t_crop',
    #                     default='random',
    #                     type=str,
    #                     help=('Temporal cropping method in training. '
    #                           'random is uniform. '
    #                           '(random | center)'))
    # parser.add_argument('--learning_rate',
    #                     default=0.1,
    #                     type=float,
    #                     help=('Initial learning rate'
    #                           '(divided by 10 while training by lr scheduler)'))
    # parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    # parser.add_argument('--dampening',
    #                     default=0.0,
    #                     type=float,
    #                     help='dampening of SGD')
    # parser.add_argument('--weight_decay',
    #                     default=1e-3,
    #                     type=float,
    #                     help='Weight Decay')
    # parser.add_argument('--mean_dataset',
    #                     default='kinetics',
    #                     type=str,
    #                     help=('dataset for mean values of mean subtraction'
    #                           '(activitynet | kinetics | 0.5)'))
    # parser.add_argument('--no_mean_norm',
    #                     action='store_true',
    #                     help='If true, inputs are not normalized by mean.')
    # parser.add_argument(
    #     '--no_std_norm',
    #     action='store_true',
    #     help='If true, inputs are not normalized by standard deviation.')
    # parser.add_argument(
    #     '--value_scale',
    #     default=1,
    #     type=int,
    #     help=
    #     'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    # parser.add_argument('--nesterov',
    #                     action='store_true',
    #                     help='Nesterov momentum')
    # parser.add_argument('--optimizer',
    #                     default='sgd',
    #                     type=str,
    #                     help='Currently only support SGD')
    # parser.add_argument('--lr_scheduler',
    #                     default='multistep',
    #                     type=str,
    #                     help='Type of LR scheduler (multistep | plateau)')
    # parser.add_argument(
    #     '--multistep_milestones',
    #     default=[50, 100, 150],
    #     type=int,
    #     nargs='+',
    #     help='Milestones of LR scheduler. See documentation of MultistepLR.')
    # parser.add_argument(
    #     '--overwrite_milestones',
    #     action='store_true',
    #     help='If true, overwriting multistep_milestones when resuming training.'
    # )
    # parser.add_argument(
    #     '--plateau_patience',
    #     default=10,
    #     type=int,
    #     help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    # )
    # parser.add_argument('--batch_size',
    #                     default=128,
    #                     type=int,
    #                     help='Batch Size')
    # parser.add_argument(
    #     '--inference_batch_size',
    #     default=0,
    #     type=int,
    #     help='Batch Size for inference. 0 means this is the same as batch_size.'
    # )
    # parser.add_argument(
    #     '--batchnorm_sync',
    #     action='store_true',
    #     help='If true, SyncBatchNorm is used instead of BatchNorm.')
    # parser.add_argument('--n_epochs',
    #                     default=200,
    #                     type=int,
    #                     help='Number of total epochs to run')
    # parser.add_argument('--n_val_samples',
    #                     default=3,
    #                     type=int,
    #                     help='Number of validation samples for each activity')
    # parser.add_argument('--resume_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Save data (.pth) of previous training')
    # parser.add_argument('--no_train',
    #                     action='store_true',
    #                     help='If true, training is not performed.')
    # parser.add_argument('--no_val',
    #                     action='store_true',
    #                     help='If true, validation is not performed.')
    # parser.add_argument('--inference',
    #                     action='store_true',
    #                     help='If true, inference is performed.')
    # parser.add_argument('--inference_subset',
    #                     default='val',
    #                     type=str,
    #                     help='Used subset in inference (train | val | test)')
    # parser.add_argument('--inference_stride',
    #                     default=16,
    #                     type=int,
    #                     help='Stride of sliding window in inference.')
    # parser.add_argument(
    #     '--inference_crop',
    #     default='center',
    #     type=str,
    #     help=('Cropping method in inference. (center | nocrop)'
    #           'When nocrop, fully convolutional inference is performed,'
    #           'and mini-batch consists of clips of one video.'))
    # parser.add_argument(
    #     '--inference_no_average',
    #     action='store_true',
    #     help='If true, outputs for segments in a video are not averaged.')
    # parser.add_argument('--no_cuda',
    #                     action='store_true',
    #                     help='If true, cuda is not used.')
    # parser.add_argument('--n_threads',
    #                     default=4,
    #                     type=int,
    #                     help='Number of threads for multi-thread loading')
    # parser.add_argument('--checkpoint',
    #                     default=10,
    #                     type=int,
    #                     help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth',
                        default=18,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    # parser.add_argument('--wide_resnet_k',
    #                     default=2,
    #                     type=int,
    #                     help='Wide resnet k')
    # parser.add_argument('--resnext_cardinality',
    #                     default=32,
    #                     type=int,
    #                     help='ResNeXt cardinality')
    # parser.add_argument('--input_type',
    #                     default='rgb',
    #                     type=str,
    #                     help='(rgb | flow)')
    # parser.add_argument('--manual_seed',
    #                     default=1,
    #                     type=int,
    #                     help='Manually set random seed')
    # parser.add_argument('--accimage',
    #                     action='store_true',
    #                     help='If true, accimage is used to load images.')
    # parser.add_argument('--output_topk',
    #                     default=5,
    #                     type=int,
    #                     help='Top-k scores are saved in json file.')
    # parser.add_argument('--file_type',
    #                     default='jpg',
    #                     type=str,
    #                     help='(jpg | hdf5)')
    # parser.add_argument('--tensorboard',
    #                     action='store_true',
    #                     help='If true, output tensorboard log file.')
    # parser.add_argument(
    #     '--distributed',
    #     action='store_true',
    #     help='Use multi-processing distributed training to launch '
    #     'N processes per node, which has N GPUs.')
    # parser.add_argument('--dist_url',
    #                     default='tcp://127.0.0.1:23456',
    #                     type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--world_size',
    #                     default=-1,
    #                     type=int,
    #                     help='number of nodes for distributed training')

    args = parser.parse_args()

    return args
