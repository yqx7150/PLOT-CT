from torch.utils import data as data
from torchvision.transforms.functional import normalize
import os
import cv2
from DiffIR.data.data_util import (paired_paths_from_folder,
                                   paired_DP_paths_from_folder,
                                   paired_paths_from_lmdb,
                                   paired_paths_from_meta_info_file)
from DiffIR.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from DiffIR.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
import random
import numpy as np
import torch
import odl
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.morphology import dilation, erosion
from scipy.ndimage import binary_fill_holes
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,
                                        src_radius=500, det_radius=500)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)
mu_water = 0.02
epsilon = 0.0001
nonlinear_operator = odl.ufunc_ops.exp(Fan_ray_trafo.range) * (-mu_water * Fan_ray_trafo)
photons = 1e4


# get noise
def init_ct_op(img):
    global photons
    photons_per_pixel = photons
    nonlinear_operator = odl.ufunc_ops.exp(Fan_ray_trafo.range) * (-mu_water * Fan_ray_trafo)
    phantom = Fan_reco_space.element(img)
    # phantom = phantom / 1000.0
    # maxdegrade = np.max(phantom)
    # phantom = phantom/maxdegrade
    proj_trans = Fan_ray_trafo(phantom)
    # proj_trans = padding_img(proj_trans)
    proj_data = nonlinear_operator(phantom)
    # proj_data = padding_img(proj_data)
    proj_ideal = np.copy(proj_trans)
    proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel) / photons_per_pixel
    # proj_data = -np.log(epsilon + proj_data) / mu_water
    # proj_data = proj_data * maxdegrade

    # maxvalue1 = np.max(proj_data)
    # minvalue1 = np.min(proj_data)
    # proj_data = (proj_data - minvalue1) / (maxvalue1 - minvalue1)
    '''
  mean = 0
  var = 0.5
  phantom = Fan_reco_space.element(img)
  proj_trans = Fan_ray_trafo(phantom)
  proj_trans = img_as_float(proj_trans)
  noise = np.random.normal(mean, var**0.55, proj_trans.shape)
  proj_data = proj_trans + noise
  proj_data = np.clip(proj_data, 0.0, 255.0)
  plt.imshow(proj_data, 'gray')
  plt.show()
  '''
    sinogram_input = proj_data
    sinogram_input = sinogram_input.asarray()

    img_ldct = Fan_FBP(sinogram_input)
    '''
  #plt.imshow(sinogram_input, 'gray')
  #plt.show()
  #plt.imshow(phantom, 'gray')
  #plt.show()
  plt.imshow(img_ldct, 'gray')
  plt.show()
  assert 0
  '''
    return sinogram_input


def img_to_sinogram(img):
    return Fan_ray_trafo(img)


@DATASET_REGISTRY.register()
class DeblurPairedDataset(data.Dataset):
    """Paired image dataset for image restoration with sinogram saving"""

    def __init__(self, opt):
        super(DeblurPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.save_dir = 'saved_sinograms'
        self.save_gt_dir = os.path.join(self.save_dir, 'gt_sinograms')
        os.makedirs(self.save_gt_dir, exist_ok=True)
        self.gt_class_dirs = [
            os.path.join(self.save_gt_dir, f'class{i + 1}') for i in range(4)
        ]
        for dir_path in self.gt_class_dirs:
            os.makedirs(dir_path, exist_ok=True)

        self.save_lq_dir = os.path.join(self.save_dir, 'lq_sinograms')
        os.makedirs(self.save_lq_dir, exist_ok=True)
        self.lq_class_dirs = [
            os.path.join(self.save_lq_dir, f'class{i + 1}') for i in range(4)
        ]
        for dir_path in self.lq_class_dirs:
            os.makedirs(dir_path, exist_ok=True)

        # Path loading logic remains unchanged
        if self.io_backend_opt['type'] == 'lmdb':
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file']:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        self.geometric_augs = opt['geometric_augs'] if opt['phase'] == 'train' else False

    def padding_img(self, img):
        w, h = img.shape
        h1 = (h // 64 + 1) * 64
        tmp = np.zeros([h1, h1])
        x_start = int((h1 - w) // 2)
        y_start = int((h1 - h) // 2)
        tmp[x_start:x_start + w, y_start:y_start + h] = img
        return tmp

    def img_normalized(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Prevent division by zero

    def perform_clustering(self, sinogram_gt, sinogram_lq, n_clusters, fname):
        """
        Perform K-Means clustering and save images with original file names
        :param sinogram_gt: GT sinogram data
        :param sinogram_lq: LQ sinogram data
        :param n_clusters: Number of clusters
        :param fname: Original file name (without extension)
        :return: List of clustered GT and LQ images
        """
        kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, init='k-means++', random_state=42)
        reshaped_sinogram = sinogram_gt.reshape(-1, 1)
        kmeans.fit(reshaped_sinogram)
        labels = kmeans.labels_.reshape(sinogram_gt.shape)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        # print(sorted_indices)
        # assert 0
        sinogram_gt_temp = np.clip(self.img_normalized(sinogram_gt) * 255, 0, 255)
        sinogram_lq_temp = np.clip(self.img_normalized(sinogram_lq) * 255, 0, 255)
        re_sinogram_gt = []
        re_sinogram_lq = []

        for i, cluster_id in enumerate(sorted_indices):
            mask = (labels == cluster_id).astype(np.float32)
            cluster_image_gt_temp = (sinogram_gt_temp * mask).astype(np.float32)
            cluster_image_lq_temp = (sinogram_lq_temp * mask).astype(np.float32)
            cluster_image_gt = (sinogram_gt * mask).astype(np.float32)
            cluster_image_lq = (sinogram_lq * mask).astype(np.float32)
            # Save clustered GT and LQ results to corresponding folders
            # if True:  # Use the passed file name for judgment and saving
            #     cv2.imwrite(
            #         f'/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/saved_sinograms/ori_sinogram/{fname}.png',
            #         sinogram_gt_temp)
            #     # Save GT clustered image
            #     gt_save_path = os.path.join(self.gt_class_dirs[i], f'{fname}_class{i + 1}.png')
            #     cv2.imwrite(gt_save_path, cluster_image_gt_temp)
            #
            #     # Save LQ clustered image
            #     lq_save_path = os.path.join(self.lq_class_dirs[i], f'{fname}_class{i + 1}.png')
            #     cv2.imwrite(lq_save_path, cluster_image_lq_temp)

            re_sinogram_gt.append(cluster_image_gt)
            re_sinogram_lq.append(cluster_image_lq)

        return re_sinogram_gt, re_sinogram_lq

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        original_index = index
        index = index % len(self.paths)
        item = self.paths[index]
        gt_path = item['gt_path']
        lq_path = item['lq_path']

        # Extract original file name (without extension)
        gt_filename = os.path.splitext(os.path.basename(gt_path))[0]
        lq_filename = os.path.splitext(os.path.basename(lq_path))[0]
        assert gt_filename == lq_filename, "GT and LQ file names must be consistent"
        fname = gt_filename  # Use unified file name

        sinogram_gt = np.load(gt_path)
        # sinogram_gt = self.padding_img(sinogram_gt)

        sinogram_lq = np.load(lq_path)
        # sinogram_lq = self.padding_img(sinogram_lq)
        # sinogram_gt,sinogram_lq = self.img_normalized(sinogram_gt),self.img_normalized(sinogram_lq)

        # sinogram_gt,sinogram_lq=padding(sinogram_gt,sinogram_lq,gt_size)

        # Call clustering function and pass the file name
        sinogram_gt, sinogram_lq = self.perform_clustering(sinogram_gt, sinogram_lq, 3, fname)

        # Convert to NumPy array (assuming clustering returns a list)
        # sinogram_gt = sinogram_gt[None, :, :]
        # sinogram_lq = sinogram_lq[None, :, :]
        sinogram_gt = np.array(sinogram_gt)
        sinogram_lq = np.array(sinogram_lq)
        # print(sinogram_gt.shape, sinogram_lq.shape)
        # assert 0
        # print("Clustered shape:", sinogram_gt.shape, sinogram_lq.shape)
        # sinogram_gt = np.transpose(sinogram_gt, (1, 2, 0))
        # temp = np.sum(sinogram_gt, axis=0)
        # temp = np.array(Fan_FBP(temp))
        #
        # # Save original floating-point matrix (retain all decimals and negative values)
        # np.save('./temp.npy', temp)
        #
        # # Generate visualization grayscale image if needed (loses decimal information)
        # normalized = (temp - temp.min()) / (temp.max() - temp.min()) * 255
        # cv2.imwrite('./temp_visualization.png', normalized.astype(np.uint8))
        #
        # assert 0
        # Data augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Adjust dimension order: (classes, height, width) -> (height, width, classes)
            # sinogram_gt, sinogram_lq = padding(sinogram_gt, sinogram_lq, gt_size)
            sinogram_gt = np.transpose(sinogram_gt, (1, 2, 0))
            sinogram_lq = np.transpose(sinogram_lq, (1, 2, 0))
            sinogram_gt, sinogram_lq = paired_random_crop(
                sinogram_gt, sinogram_lq, gt_size, self.opt['scale'], gt_path
            )
            if self.geometric_augs:
                sinogram_gt, sinogram_lq = random_augmentation(sinogram_gt, sinogram_lq)
        if self.opt['phase'] == 'val':
            # gt_size = self.opt['gt_size']
            # Adjust dimension order: (classes, height, width) -> (height, width, classes)
            # sinogram_gt, sinogram_lq = padding(sinogram_gt, sinogram_lq, gt_size)
            sinogram_gt = np.transpose(sinogram_gt, (1, 2, 0))
            sinogram_lq = np.transpose(sinogram_lq, (1, 2, 0))
            # sinogram_gt, sinogram_lq = paired_random_crop(
            #     sinogram_gt, sinogram_lq, gt_size, self.opt['scale'], gt_path
            # )
            # if self.geometric_augs:
            #     sinogram_gt, sinogram_lq = random_augmentation(sinogram_gt, sinogram_lq)

        # Convert to Tensor (CHW format)
        sinogram_gt = img2tensor(sinogram_gt, float32=True)  # HWC -> CHW
        sinogram_lq = img2tensor(sinogram_lq, float32=True)

        # Normalization
        if self.mean is not None or self.std is not None:
            normalize(sinogram_lq, self.mean, self.std, inplace=True)
            normalize(sinogram_gt, self.mean, self.std, inplace=True)
        return {
            'lq': sinogram_lq,
            'gt': sinogram_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def save_with_plt(self, image, save_dir, filename, class_name):
        """Save images using matplotlib with automatic normalization"""
        # Generate full path
        save_path = os.path.join(save_dir, f'{filename}_{class_name}.png')
        # Save as grayscale image
        plt.imsave(save_path, image, cmap='gray')
        plt.close()  # Close the image to free memory

    def __len__(self):
        return len(self.paths)


class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # Random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # Flip and rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # Random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)

            # Flip and rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                              bgr2rgb=True,
                                              float32=True)
        # Normalization
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)