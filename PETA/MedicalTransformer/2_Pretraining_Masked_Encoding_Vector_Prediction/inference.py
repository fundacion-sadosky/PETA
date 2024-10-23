import torch.optim as optim
from models.ResNet_Model import Multiview_MEP
from losses import *
from helpers import *
import torch
import torch.nn.functional as F
import nibabel as nib
import nilearn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from nilearn import plotting
import torchvision
from torchvision import transforms, utils, models, datasets
from multiprocessing import Pool
import time

import sys
sys.path.append('../../src')
from transforms import ToLabelOutput, TransformGridImage, TransformReduced3DImage, Transform3DImage, MinMaxNormalization
from datasets import ADNIDataset
from dl_builder import DLBuilder
from datasets_pretrain import PretrainDatasetBuilder
from cross_validation import getKFoldTrainAndValDatasets
from classification_lib import Multiview_Classification

def show_results(img, img_np_tensor, sensitivity_map):
    # sensitivity_map[sensitivity_map < 0] = 0
    # sensitivity_map = np.abs(sensitivity_map)
    # sensitivity_map = sensitivity_map * -1
    if sensitivity_map.max() != sensitivity_map.min():
        # Normalize sensitivity map if needed
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min())

    print(sensitivity_map)

    # Visualize or further analyze the sensitivity map as desired


    # Assuming sensitivity_map is normalized, if not, normalize it before visualization

    cut_coords = max_index = np.unravel_index(np.argmax(sensitivity_map), sensitivity_map.shape)
    print("Cut coords:")
    print(cut_coords)
    print(sensitivity_map[cut_coords[0], cut_coords[1], cut_coords[2]])

    # Plot the original 3D image
    img_np_normalized = (img_np_tensor - img_np_tensor.min()) / (img_np_tensor.max() - img_np_tensor.min())
    print(img_np_tensor.shape)
    plotting.plot_img(nilearn.image.new_img_like(img, img_np_normalized, affine = img.affine), title='Original 3D Image', cmap='gray', cut_coords = cut_coords)

    print("sensitivity map shape")
    print(sensitivity_map.shape)
    print(sensitivity_map.min())
    print(sensitivity_map.max())
    print(sensitivity_map.mean())
    print(sensitivity_map.std())
    print(np.unique(sensitivity_map))

    print("Min/max de np tensor:")
    print(img_np_normalized.min())
    print(img_np_normalized.max())
    

    bg_img = nilearn.image.new_img_like(img, img_np_normalized, affine = img.affine)
    # bg_img = None
    
    # img_np_tensor.astype('float32')
    # Plot the sensitivity map as a heatmap on top of the original image
    plotting.plot_roi(nilearn.image.new_img_like(img, sensitivity_map * 255), bg_img=bg_img, cmap='plasma', colorbar=True, title='Sensitivity Map', cut_coords = cut_coords)

    # Show the plots
    plt.show()

def run(args):
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load your pre-trained PyTorch model
    model = Multiview_Classification(args, device).to(device)

    model_dict = torch.load(args.weights, map_location=device)
    
    # Remove the "module." prefix from keys
    # Esto es agregado por DataParallel
    new_state_dict = {}
    for key, value in model_dict.items():
        new_key = key.replace('module.', '')  # Remove the "module." prefix
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load 3D NIfTI image
    img_path = args.img
    img = nib.load(img_path)

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
        transform3dImage = TransformReduced3DImage()
    else:
        transform3dImage = Transform3DImage(yDim = args.axial_slicelen, augmentation = 'no', resampleZ = args.adni_resamplez)

    inferenceTransform = torchvision.transforms.Compose([
        transform3dImage,
        torchvision.transforms.ToTensor(),
        normalization
    ])

    metadata = {"deleteIndices": []} # fake metadata so it works
    img_tensor = inferenceTransform([img, metadata])
    print(img_tensor.shape)

    img_tensor = img_tensor.to(device)
    
    # Define occlusion parameters
    occlusion_size = args.occlusion_size
    stride = args.occlusion_stride

    # Make a prediction on the original image
    with torch.no_grad():
        original_prediction = model(img_tensor.unsqueeze(0)).to(device)
        print("original_prediction:")
        print(torch.nn.functional.softmax(original_prediction))

    if not args.calculate_heatmap:
        sys.exit(1)

    img_np_tensor = img_tensor.cpu().numpy()

    # Initialize sensitivity map
    sensitivity_map = np.zeros_like(img_np_tensor)
    print(sensitivity_map)

    if args.expected_class == 1:
        target_labels = torch.tensor([0, 1], dtype = torch.float32).unsqueeze(0).to(device)
    else:
        target_labels = torch.tensor([1, 0], dtype = torch.float32).unsqueeze(0).to(device)
    
    # Occlusion loop for 3D image
    occlusion_args = []
    for i in range(0, img_np_tensor.shape[0] - occlusion_size + 1, stride):
        for j in range(0, img_np_tensor.shape[1] - occlusion_size + 1, stride):
            for k in range(0, img_np_tensor.shape[2] - occlusion_size + 1, stride):
                occlusion_args.append((i, j, k, original_prediction, target_labels, model, img_tensor, occlusion_size, device, args.expected_class))

    print(f"Total occlusions to be calculated: {len(occlusion_args)}")

    calculated_occlusions = 0

    start = time.time()

    if torch.cuda.is_available() or args.num_workers == 1:
        print("Not doing multiple workers b/c it's CUDA or num_workers == 1")
        for occlusion_arg in occlusion_args:
            i, j, k, impact, _ = calculate_occlusion_wrapper(occlusion_arg)
            sensitivity_map[i:i+occlusion_size, j:j+occlusion_size, k:k+occlusion_size] += impact
            # print(f"Processed result for location ({i}, {j}, {k})")
            calculated_occlusions += 1
            # remaining_time = round(time_to_calculate * (len(occlusion_args) - calculated_occlusions) / args.num_workers)
            # Calculamos el remaining time de acuerdo a las oclusiones calculadas
            occlusions_to_calculate = len(occlusion_args) - calculated_occlusions
            elapsed_time = time.time() - start
            remaining_time = round( occlusions_to_calculate * elapsed_time / calculated_occlusions / 60 )
            print(f"Occlusions calculated: {calculated_occlusions}/{len(occlusion_args)}. Remaining time: {remaining_time} minutes")
    else:
        print("Multiprocess processing because it's CPU")
        # Create a pool of worker processes
        with Pool(args.num_workers) as pool:
            imap_results = pool.imap(calculate_occlusion_wrapper, occlusion_args)

            # Process results in real-time
            for result in imap_results:
                i, j, k, impact, _ = result
                sensitivity_map[i:i+occlusion_size, j:j+occlusion_size, k:k+occlusion_size] += impact
                # print(f"Processed result for location ({i}, {j}, {k})")
                calculated_occlusions += 1
                # remaining_time = round(time_to_calculate * (len(occlusion_args) - calculated_occlusions) / args.num_workers)
                # Calculamos el remaining time de acuerdo a las oclusiones calculadas
                occlusions_to_calculate = len(occlusion_args) - calculated_occlusions
                elapsed_time = time.time() - start
                remaining_time = round( occlusions_to_calculate * elapsed_time / calculated_occlusions / 60 )
                print(f"Occlusions calculated: {calculated_occlusions}/{len(occlusion_args)}. Remaining time: {remaining_time} minutes")

    print("Final sensitivity map:")
    print(sensitivity_map)

    print("Final max:")
    print(sensitivity_map.max())

    if args.save_heatmap:
        print("Guardando heatmap")
        np.save(f"./{args.save_heatmap}_original.npy", img_tensor.detach().cpu().numpy())
        np.save(f"./{args.save_heatmap}_heatmap.npy", sensitivity_map)

def calculate_occlusion(i, j, k, original_prediction, target_labels, model, img_tensor, occlusion_size, device, expected_class):
    start = time.time()
    print(f"Occlusion {i}/{j}/{k}")
    # Create occluded image
    occluded_img = img_tensor.clone()
    occluded_img[i:i+occlusion_size, j:j+occlusion_size, k:k+occlusion_size] = -1  # Occlude the region with -1
    occluded_tensor = occluded_img.unsqueeze(0).float().to(device)

    # Make prediction on occluded image
    with torch.no_grad():
        occluded_prediction = model(occluded_tensor).to(device)

    # Compute cross-entropy loss on both predictions
    original_loss = F.cross_entropy(original_prediction, target_labels)
    occluded_loss = F.cross_entropy(occluded_prediction, target_labels)
    # print("original vs occluded loss:")
    # print(original_loss)
    # print(occluded_loss)

    original_prediction = torch.nn.functional.softmax(original_prediction)
    occluded_prediction = torch.nn.functional.softmax(occluded_prediction)

    # print("Origianl prediction")
    # print(original_prediction)
    # print("Occluded prediction:")
    # print(occluded_prediction)

    # Measure impact (change in cross-entropy loss)
    # Queremos subrayar las regiones donde cambia la loss como positivas
    # Y donde cambia menos como negativas
    # impact = original_loss.item() - occluded_loss.item() # Usar math.abs? Poner en 0? Me importa si al tapar algo ME DISMINUYE la loss?

    # Si son 3 clases esto cambiaría, pero teniendo solo 2 clases la predicción de AD es suficiente
    if expected_class == 1:
        impact = (original_prediction[0][1] - occluded_prediction[0][1]).item()
    else:
        impact = (original_prediction[0][0] - occluded_prediction[0][0]).item()
    
    # print("impact")
    # print(impact)
    
    end = time.time()
    # print(f"Time to calculate 1 occlusion: {end - start}")

    time_to_calculate = end - start
    
    return i, j, k, impact, time_to_calculate

def calculate_occlusion_wrapper(args):
    i, j, k, impact, time_to_calculate = calculate_occlusion(*args)
    return i, j, k, impact, time_to_calculate

if __name__ == '__main__':
    # Define Arguments
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--pre_dataset", type=str, default='ixi_camcan_abide')
    parser.add_argument("--model", type=str, default='7_Multiview_MEP_CN_ResNet_freeze')
    
    parser.add_argument("--is_pool", type=int, default=1)
    parser.add_argument("--normalization", type=str, default="min-max")

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

    parser.add_argument("--weights", type=str)
    parser.add_argument("--img", type=str)


    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-intensity", type=float, default = -1.0)

    parser.add_argument("--fc-layers", type=int, default=2)

    parser.add_argument("--collapse-ydim", type=float, default = 0.0)

    parser.add_argument("--adni-resamplez", type=float, default = None)
    parser.add_argument("--fleni-resamplez", type=float, default=0.610389)

    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--calculate-heatmap", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-heatmap", type=str, default=None)

    parser.add_argument("--occlusion-size", type=int, default = 40)
    parser.add_argument("--occlusion-stride", type=int, default=20)
    parser.add_argument("--expected-class", type=int, default=1)
    
    args = parser.parse_args()

    run(args)
