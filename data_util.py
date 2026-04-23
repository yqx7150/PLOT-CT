import os
import numpy as np
import random
import shutil
import odl
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.morphology import dilation, erosion
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.ndimage import uniform_filter

# Set random seed
random.seed(42)
np.random.seed(42)

# Dataset root directory
DATASET_ROOT = './test1e4'
VISUALIZATION_FOLDER = os.path.join(DATASET_ROOT, 'sinogram_visualizations')  # Visualization folder
IMG_VISUALIZATION_FOLDER = os.path.join(DATASET_ROOT, 'img_visualizations')  # Noisy image visualization folder

# Define ODL space and transform
Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition, src_radius=500, det_radius=500)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)  # img2sinogram
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)

# Noise addition parameters
photons = [1e4]
epsilon = 0.0001
mu_water = 0.02

# Specify the list of filenames to process
TARGET_FILES = ['1', '2', '3', '4', '5', '6', '7', '8',
                '9', '10', '11', '12']


# TARGET_FILES = ['2279', '2403', '2838', '2924', '3149', '3237', '3257', '3330',
#                 '3770', '4150', '4328', '2893']

def img_normalized(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Prevent division by zero


def init_ct_op(img, photon):
    """Add CT scan noise to the input image and return noisy image and sinogram"""
    phantom = Fan_reco_space.element(img)
    proj_data = Fan_ray_trafo(phantom)  # Original projection data

    # Add noise
    nonlinear_operator = odl.ufunc_ops.exp(Fan_ray_trafo.range) * (-mu_water * Fan_ray_trafo)
    noisy_proj = nonlinear_operator(phantom)
    pre_sinogram = np.array(noisy_proj)
    noisy_proj = odl.phantom.poisson_noise(noisy_proj * photon) / photon
    sinogram_input1 = (-np.log(epsilon + noisy_proj)) / mu_water
    sinogram_input = noisy_proj

    # Use FBP reconstruction as noisy image
    img_ldct = Fan_FBP(sinogram_input)
    img_ldct1 = Fan_FBP(sinogram_input1)
    return img_ldct.asarray(), img_ldct1.asarray(), sinogram_input.asarray(), pre_sinogram, proj_data, sinogram_input1  # Return noisy image and sinogram


def save_visualization_opencv(image, path):
    """Save image using OpenCV (automatically normalize and convert to uint8 format)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Normalize to [0, 255] range and convert to uint8
    img_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)) * 255
    img_uint8 = img_normalized.astype(np.uint8)

    # Save as grayscale image (single channel)
    cv2.imwrite(path, img_uint8)


def check_image_size(image, filename, target_size=(720, 720)):
    """Check if image size meets requirements"""
    if image.shape != target_size:
        raise ValueError(f"Error: {filename} has incorrect size ({image.shape} != {target_size})")


def copy_dataset_to_val(train_dir, val_dir):
    """Copy train dataset to val directory (with error handling for existing directories)"""
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through subfolders
    for folder in ['full', 'low']:
        src = os.path.join(train_dir, folder)
        dst = os.path.join(val_dir, folder)

        # Check if source directory exists
        if not os.path.exists(src):
            print(f"Warning: Source directory {src} does not exist, skipping...")
            continue

        # If destination exists, remove it first to avoid shutil error
        if os.path.exists(dst):
            print(f"Destination directory {dst} already exists, removing it first...")
            shutil.rmtree(dst)

        # Copy directory with progress indication
        try:
            shutil.copytree(src, dst)
            print(f"Successfully copied {src} to {dst}")
        except Exception as e:
            print(f"Error copying {src} to {dst}: {str(e)}")

    print(f"Finished copying train dataset to val: {val_dir}")


def main():
    raw_full_folder = './datasets'  # Original image folder
    val1_full_folder = os.path.join(DATASET_ROOT, 'val1/full')
    val1_low_folder = os.path.join(DATASET_ROOT, 'val1/low')
    val2_full_folder = os.path.join(DATASET_ROOT, 'val2/full')
    val2_low_folder = os.path.join(DATASET_ROOT, 'val2/low')

    # Create directories
    os.makedirs(val1_full_folder, exist_ok=True)
    os.makedirs(val1_low_folder, exist_ok=True)
    os.makedirs(val2_full_folder, exist_ok=True)
    os.makedirs(val2_low_folder, exist_ok=True)
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    os.makedirs(IMG_VISUALIZATION_FOLDER, exist_ok=True)  # New: folder for saving noisy images

    # Process only specified files
    npy_files = [f"{fname}.npy" for fname in TARGET_FILES]
    print(f"Processing {len(npy_files)} samples...")

    for photon in photons:
        processed_count = 0  # Counter to track number of processed images
        for i, filename in enumerate(tqdm(npy_files)):
            fname = os.path.splitext(filename)[0]

            # Check if file exists
            img_path = os.path.join(raw_full_folder, filename)
            if not os.path.exists(img_path):
                print(f"Warning: File {filename} does not exist, skipping...")
                continue

            print(f'Processing file {filename}')
            img = np.load(img_path)
            img = img_normalized(img)

            # Generate clean sinogram
            sinogram_gt = Fan_ray_trafo(Fan_reco_space.element(img)).asarray()

            # Generate noisy image and noisy sinogram
            img_lq, x, sinogram_lq, sinogram_pre, sinogram_post, sinogram_post_log = init_ct_op(img, photon)

            # Check image size
            check_image_size(sinogram_pre, f"{filename}_gt_sinogram")
            check_image_size(sinogram_lq, f"{filename}_lq_sinogram")

            # Save data files
            np.save(os.path.join(val1_full_folder, f'{fname}_{photon}.npy'), sinogram_pre)
            np.save(os.path.join(val1_low_folder, f'{fname}_{photon}.npy'), sinogram_lq)

            processed_count += 1  # Increment processing counter

            # Visualize all processed images
            gt_viz_path = os.path.join(VISUALIZATION_FOLDER, f"{fname}_gt_sinogram.png")
            plt.imsave(gt_viz_path, sinogram_pre, cmap='gray')

            lq_sinogram_viz_path = os.path.join(VISUALIZATION_FOLDER, f"{fname}_lq_sinogram.png")
            plt.imsave(lq_sinogram_viz_path, sinogram_lq, cmap='gray')

            # Save noisy image visualization (OpenCV grayscale)
            lq_img_viz_path = os.path.join(IMG_VISUALIZATION_FOLDER, f"{fname}_lq_img.png")
            plt.imsave(lq_img_viz_path, x, cmap='gray')
            # plt.imsave(f'/home/un/world/yx/1_CHAOS/No_reconstruction/img/png/1e4/{fname}_10000.0.png', x, cmap='gray')
            # np.save(f'/home/un/world/yx/1_CHAOS/No_reconstruction/img/npy/1e4/{fname}_10000.0.npy', x)

            gt_img_viz_path = os.path.join(IMG_VISUALIZATION_FOLDER, f"{fname}_gt_img.png")
            plt.imsave(gt_img_viz_path, img, cmap='gray')

        print(f"Processed {processed_count} images")
        print("Data processing completed, starting to copy to val set...")

        # Execute copy operation with clear path indication
        val1_dir = os.path.join(DATASET_ROOT, 'val1')
        val2_dir = os.path.join(DATASET_ROOT, 'val2')
        copy_dataset_to_val(val1_dir, val2_dir)

        print("All operations completed successfully!")


if __name__ == "__main__":
    main()