## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import deblur_utils as utils

from natsort import natsorted
from glob import glob
from kdsrgan.archs.TA_arch import BlindSR_TA
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/test_datasets/val1/low', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/results', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/experiments/train_DiffIRS2/models/net_g_90000.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='HIDE', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

####### Load yaml #######
yaml_file = '/home/un/world/yx/DiffIR-master/DiffIR-demotionblur/options/test_DiffIRS2.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
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


x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = BlindSR_TA(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
gt_dir = os.path.join(args.input_dir, 'test', dataset, 'target')
inp_files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
gt_files = natsorted(glob(os.path.join(gt_dir, '*.png')) + glob(os.path.join(gt_dir, '*.jpg')))

with torch.no_grad():
    for inp_file_,gt_file_ in tqdm(zip(inp_files,gt_files)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        img = np.float32(utils.load_img(gt_file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        gt_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        gt_ = F.pad(gt_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_,gt_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_file_)[-1])[0]+'.png')), img_as_ubyte(restored))
