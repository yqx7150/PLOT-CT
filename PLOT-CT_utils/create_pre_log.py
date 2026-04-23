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
DATASET_ROOT = '/home/un/world/yx/ncsnpp_ct_small_versioin/11'
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
photons = [1e3]
epsilon = 0.0001
mu_water = 0.02


def img_normalized(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # 防止除零


def init_ct_op(img,photon):
    """对输入图像添加CT扫描噪声并返回加噪图像和sinogram"""
    phantom = Fan_reco_space.element(img)
    proj_data = Fan_ray_trafo(phantom)  # 原始投影数据

    # 添加噪声
    nonlinear_operator = odl.ufunc_ops.exp(Fan_ray_trafo.range) * (-mu_water * Fan_ray_trafo)
    noisy_proj = nonlinear_operator(phantom)
    pre_sinogram = np.array(noisy_proj)
    noisy_proj = odl.phantom.poisson_noise(noisy_proj * photon) / photon
    sinogram_input1 = (-np.log(epsilon + noisy_proj)) / mu_water
    sinogram_input = noisy_proj

    # 使用FBP重建作为加噪图像
    img_ldct = Fan_FBP(sinogram_input1)
    return img_ldct.asarray(), sinogram_input.asarray(),pre_sinogram  # 返回加噪图像和sinogram


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
    raw_full_folder = '/home/un/world/yx/ncsnpp_ct_small_versioin/train'  # 原始图像文件夹
    val1_full_folder = os.path.join(DATASET_ROOT, 'train/full')
    val1_low_folder = os.path.join(DATASET_ROOT, 'train/low')
    val2_full_folder = os.path.join(DATASET_ROOT, 'val/full')
    val2_low_folder = os.path.join(DATASET_ROOT, 'val/low')

    os.makedirs(val1_full_folder, exist_ok=True)
    os.makedirs(val1_low_folder, exist_ok=True)
    os.makedirs(val2_full_folder, exist_ok=True)
    os.makedirs(val2_low_folder, exist_ok=True)
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    os.makedirs(IMG_VISUALIZATION_FOLDER, exist_ok=True)  # 新增：保存加噪图像的文件夹

    npy_files = [f for f in os.listdir(raw_full_folder) if f.endswith('.npy')]
    print(f"处理{len(npy_files)}个样本...")
    for photon in photons:
        processed_count = 0  # 计数器，用于跟踪处理的图像数量
        for i, filename in enumerate(tqdm(npy_files)):
            # 提取文件名中的数字部分
            fname = os.path.splitext(filename)[0]


            # 只处理文件编号小于等于2200的图像
            # if not fname.isdigit() or int(fname) > 2200:
            #     print(f'跳过文件 {filename}')
            #     continue
            print(f'处理文件 {filename}')

            img_path = os.path.join(raw_full_folder, filename)
            img = np.load(img_path)
            if img.shape!=(512, 512):
                print(img.shape)
                assert 0

            img = img_normalized(img)
            # 生成清晰的sinogram
            sinogram_gt = Fan_ray_trafo(Fan_reco_space.element(img)).asarray()

            # 生成加噪图像和加噪sinogram
            img_lq, sinogram_lq,sinogram_pre = init_ct_op(img,photon)

            # 尺寸检查
            check_image_size(sinogram_pre, f"{filename}_gt_sinogram")
            check_image_size(sinogram_lq, f"{filename}_lq_sinogram")

            # 保存数据文件
            np.save(os.path.join(val1_full_folder, f'{fname}_{photon}.npy'), sinogram_pre)
            np.save(os.path.join(val1_low_folder, f'{fname}_{photon}.npy'), sinogram_lq)

            processed_count += 1  # 增加处理计数

            # 可视化前5张图
            if processed_count <= 5:
                # 保存sinogram可视化（OpenCV灰度图）
                gt_viz_path = os.path.join(VISUALIZATION_FOLDER, f"{fname}_gt_sinogram.png")
                save_visualization_opencv(sinogram_pre, gt_viz_path)

                lq_sinogram_viz_path = os.path.join(VISUALIZATION_FOLDER, f"{fname}_lq_sinogram.png")
                save_visualization_opencv(sinogram_lq, lq_sinogram_viz_path)

                # 保存加噪图像可视化（OpenCV灰度图）
                lq_img_viz_path = os.path.join(IMG_VISUALIZATION_FOLDER, f"{fname}_lq_img.png")
                save_visualization_opencv(img_lq, lq_img_viz_path)

                gt_img_viz_path = os.path.join(IMG_VISUALIZATION_FOLDER, f"{fname}_gt_img.png")
                save_visualization_opencv(img, gt_img_viz_path)

        #print(f"已处理{processed_count}张图像（文件编号>2200）")
        print("数据处理完成，开始复制到val集...")
        copy_dataset_to_val(os.path.join(DATASET_ROOT, 'train'), os.path.join(DATASET_ROOT, 'val'))
        print("全部完成！")


if __name__ == "__main__":
    main()