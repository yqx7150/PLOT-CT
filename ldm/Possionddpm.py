import logging
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
from tqdm import tqdm

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.util2 import make_beta_schedule, extract_into_tensor, noise_like
from ldm.ddim import DDIMSampler


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(nn.Module):
    # classic DDPM with Gaussian-Poisson mixed diffusion, in image space
    def __init__(self,
                 denoise,
                 condition,
                 timesteps=1000,
                 beta_schedule="linear",
                 image_size=256,
                 n_feats=128,
                 clip_denoised=False,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="x0",  # all assuming fixed variance schedules
                 poisson_scale=1.0,  # 控制泊松噪声的强度
                 poisson_ratio=0.8,  # 泊松噪声在混合噪声中的比例
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode with Gaussian-Poisson mixed noise (Poisson ratio: {poisson_ratio})")
        # self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.image_size = image_size  # try conv?
        self.channels = n_feats
        self.model = denoise
        self.condition = condition
        self.poisson_scale = poisson_scale  # 泊松噪声缩放因子
        self.poisson_ratio = poisson_ratio  # 泊松噪声比例

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c, clip_denoised: bool):
        model_out = self.model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, model_out

    def p_sample(self, x, t, c, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, predicted_noise = self.p_mean_variance(x=x, t=t, c=c,
                                                                                  clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = 0
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean, predicted_noise

    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def generate_poisson_noise(self, x_start, scale=1.0):
        """
        生成泊松噪声
        Args:
            x_start: 原始图像张量
            scale: 控制泊松噪声强度的缩放因子
        Returns:
            poisson_noise: 泊松噪声
        """
        # 将输入调整到正数范围，因为泊松分布需要正数
        # 保存原始数据的范围信息
        x_min = x_start.min()
        x_max = x_start.max()

        if x_min < 0:
            # 如果有负数，先平移到正数范围
            shift = -x_min + 1e-5
            x_shifted = x_start + shift
        else:
            shift = 0
            x_shifted = x_start + 1e-5  # 避免零值

        # 应用缩放因子
        x_scaled = x_shifted * scale

        # 生成泊松噪声
        poisson_samples = torch.poisson(x_scaled)

        # 转换回原始尺度
        poisson_noise = (poisson_samples / scale) - x_shifted

        # 如果之前平移了，现在平移回去
        if shift > 0:
            poisson_noise = poisson_noise + shift

        return poisson_noise

    def generate_gaussian_poisson_mixed_noise(self, x_start, poisson_scale=1.0, poisson_ratio=0.8):
        """
        生成高斯-泊松混合噪声
        Args:
            x_start: 原始图像张量
            poisson_scale: 控制泊松噪声强度的缩放因子
            poisson_ratio: 泊松噪声在混合噪声中的比例（0-1）
        Returns:
            mixed_noise: 混合噪声
        """
        # 生成泊松噪声
        poisson_noise = self.generate_poisson_noise(x_start, poisson_scale)

        # 生成高斯噪声（与输入同形状的标准正态分布）
        gaussian_noise = torch.randn_like(x_start)

        # 按比例混合两种噪声
        mixed_noise = poisson_ratio * poisson_noise + (1 - poisson_ratio) * gaussian_noise

        return mixed_noise

    def q_sample(self, x_start, t, noise=None):
        """
        使用高斯-泊松混合噪声的前向扩散过程
        """
        if noise is None:
            # 生成高斯-泊松混合噪声
            noise = self.generate_gaussian_poisson_mixed_noise(
                x_start,
                poisson_scale=self.poisson_scale,
                poisson_ratio=self.poisson_ratio
            )

        # 使用标准的DDPM加噪公式
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, noise=None):
        # 使用高斯-泊松混合噪声
        if noise is None:
            noise = self.generate_gaussian_poisson_mixed_noise(
                x_start,
                poisson_scale=self.poisson_scale,
                poisson_ratio=self.poisson_ratio
            )

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        return model_out, target

    def forward(self, img, x=None):
        device = self.betas.device
        b = img.shape[0]
        if self.training:
            pred_IPR_list = []
            t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)

            # 使用高斯-泊松混合噪声
            noise = self.generate_gaussian_poisson_mixed_noise(
                x,
                poisson_scale=self.poisson_scale,
                poisson_ratio=self.poisson_ratio
            )
            x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
            IPR = x_noisy
            c = self.condition(img)

            for i in reversed(range(0, self.num_timesteps)):
                IPR, predicted_noise = self.p_sample(IPR, torch.full((b,), i, device=device, dtype=torch.long), c,
                                                     clip_denoised=self.clip_denoised)
                pred_IPR_list.append(IPR)
            return IPR, pred_IPR_list
        else:
            shape = (img.shape[0], self.channels * 4)
            x_noisy = torch.randn(shape, device=device)
            c = self.condition(img)
            IPR = x_noisy
            for i in reversed(range(0, self.num_timesteps)):
                IPR, _ = self.p_sample(IPR, torch.full((b,), i, device=device, dtype=torch.long), c,
                                       clip_denoised=self.clip_denoised)
            return IPR