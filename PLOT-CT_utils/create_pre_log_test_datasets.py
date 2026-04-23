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

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 数据集根目录
DATASET_ROOT = '/home/un/world/yx/datasets/CHAOS/test1e4'
VISUALIZATION_FOLDER = os.path.join(DATASET_ROOT, 'sinogram_visualizations')  # 可视化文件夹
IMG_VISUALIZATION_FOLDER = os.path.join(DATASET_ROOT, 'img_visualizations')  # 加噪图像可视化文件夹

# 定义ODL空间和变换
Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition, src_radius=500, det_radius=500)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)  # img2sinogram
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)

# 加噪参数
photons = [1e4]
epsilon = 0.0001
mu_water = 0.02

# 指定要处理的文件名列表
TARGET_FILES = ['1', '2', '3', '4', '5', '6', '7', '8',
                '9', '10', '11', '12']
# TARGET_FILES = ['2279', '2403', '2838', '2924', '3149', '3237', '3257', '3330',
#                 '3770', '4150', '4328', '2893']

def img_normalized(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # 防止除零


def init_ct_op(img, photon):
    """对输入图像添加CT扫描噪声并返回加噪图像和sinogram"""
    phantom = Fan_reco_space.element(img)
    proj_data = Fan_ray_trafo(phantom)  # 原始投影数据
    # print(proj_data.shape)
    # assert 0
    # 添加噪声
    nonlinear_operator = odl.ufunc_ops.exp(Fan_ray_trafo.range) * (-mu_water * Fan_ray_trafo)
    noisy_proj = nonlinear_operator(phantom)
    pre_sinogram = np.array(noisy_proj)
    noisy_proj = odl.phantom.poisson_noise(noisy_proj * photon) / photon
    sinogram_input1 = (-np.log(epsilon + noisy_proj)) / mu_water
    sinogram_input = noisy_proj

    # 使用FBP重建作为加噪图像
    img_ldct = Fan_FBP(sinogram_input)
    img_ldct1 = Fan_FBP(sinogram_input1)
    return img_ldct.asarray(), img_ldct1.asarray(),sinogram_input.asarray(), pre_sinogram,proj_data,sinogram_input1  # 返回加噪图像和sinogram


def save_visualization_opencv(image, path):
    """使用OpenCV保存图像（自动归一化并转换为uint8格式）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 归一化到[0, 255]区间并转换为uint8
    img_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)) * 255
    img_uint8 = img_normalized.astype(np.uint8)

    # 保存为灰度图像（单通道）
    cv2.imwrite(path, img_uint8)

def check_image_size(image, filename, target_size=(720, 720)):
    """检查图像尺寸是否符合要求"""
    if image.shape != target_size:
        raise ValueError(f"错误: {filename} 尺寸不正确 ({image.shape} != {target_size})")


def copy_dataset_to_val(train_dir, val_dir):
    """复制train数据集到val"""
    os.makedirs(val_dir, exist_ok=True)
    for folder in ['full', 'low']:
        src = os.path.join(train_dir, folder)
        dst = os.path.join(val_dir, folder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
    print(f"已复制train数据集到val：{val_dir}")

def main():
    raw_full_folder = '/home/un/world/yx/datasets/CHAOS_NPY'  # 原始图像文件夹
    val1_full_folder = os.path.join(DATASET_ROOT, 'val1/full')
    val1_low_folder = os.path.join(DATASET_ROOT, 'val1/low')
    val2_full_folder = os.path.join(DATASET_ROOT, 'val2/full')
    val2_low_folder = os.path.join(DATASET_ROOT, 'val2/low')

    os.makedirs(val1_full_folder, exist_ok=True)
    os.makedirs(val1_low_folder, exist_ok=True)
    os.makedirs(val2_full_folder, exist_ok=True)
    os.makedirs(val2_low_folder, exist_ok=True)
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    os.makedirs(IMG_VISUALIZATION_FOLDER, exist_ok=True)  # 新增：保存加噪图像的文件夹

    # 只处理指定的文件
    npy_files = [f"{fname}.npy" for fname in TARGET_FILES]
    # if f'{fname}'<=2200:
    print(f"处理{len(npy_files)}个样本...")

    for photon in photons:
        processed_count = 0  # 计数器，用于跟踪处理的图像数量
        for i, filename in enumerate(tqdm(npy_files)):
            fname = os.path.splitext(filename)[0]

            # 检查文件是否存在
            img_path = os.path.join(raw_full_folder, filename)
            if not os.path.exists(img_path):
                print(f"警告: 文件 {filename} 不存在，跳过")
                continue

            print(f'处理文件 {filename}')
            img = np.load(img_path)
            img = img_normalized(img)

            # 生成清晰的sinogram
            sinogram_gt = Fan_ray_trafo(Fan_reco_space.element(img)).asarray()

            # 生成加噪图像和加噪sinogram
            img_lq,x, sinogram_lq, sinogram_pre,sinogram_post,sinogram_post_log = init_ct_op(img, photon)

            # 尺寸检查
            check_image_size(sinogram_pre, f"{filename}_gt_sinogram")
            check_image_size(sinogram_lq, f"{filename}_lq_sinogram")

            # 保存数据文件
            np.save(os.path.join(val1_full_folder, f'{fname}_{photon}.npy'), sinogram_pre)
            np.save(os.path.join(val1_low_folder, f'{fname}_{photon}.npy'), sinogram_lq)

            processed_count += 1  # 增加处理计数

            # 可视化所有处理的图像
            gt_viz_path = os.path.join(VISUALIZATION_FOLDER, f"{fname}_gt_sinogram.png")
            plt.imsave(gt_viz_path, sinogram_pre, cmap='gray')

            lq_sinogram_viz_path = os.path.join(VISUALIZATION_FOLDER, f"{fname}_lq_sinogram.png")
            plt.imsave(lq_sinogram_viz_path, sinogram_lq, cmap='gray')

            #plt.imsave(f'/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/pre_test_datasets_1e4/post_log/{fname}_post_log.png', sinogram_post_log, cmap='gray')
            #plt.imsave(f'/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/pre_test_datasets_1e4/post_log/{fname}_post.png', sinogram_post, cmap='gray')

            # 保存加噪图像可视化（OpenCV灰度图）
            lq_img_viz_path = os.path.join(IMG_VISUALIZATION_FOLDER, f"{fname}_lq_img.png")
            plt.imsave(lq_img_viz_path, x, cmap='gray')
            plt.imsave(f'/home/un/world/yx/1_CHAOS/No_reconstruction/img/png/1e4/{fname}_10000.0.png', x, cmap='gray')
            np.save(f'/home/un/world/yx/1_CHAOS/No_reconstruction/img/npy/1e4/{fname}_10000.0.npy', x)

            gt_img_viz_path = os.path.join(IMG_VISUALIZATION_FOLDER, f"{fname}_gt_img.png")
            plt.imsave(gt_img_viz_path, img, cmap='gray')


        print(f"已处理{processed_count}张图像")
        print("数据处理完成，开始复制到val集...")
        copy_dataset_to_val(os.path.join(DATASET_ROOT, 'val1'), os.path.join(DATASET_ROOT, 'val2'))
        print("全部完成！")


if __name__ == "__main__":
    main()