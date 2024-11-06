from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    Losses,
)
from .sampling_config import apply_conditioning

# 定义一个用于存储样本信息的命名元组，包括轨迹（trajectories）、值（values）和链（chains）
Sample = namedtuple("Sample", "trajectories values chains")

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    """
    默认的采样函数，用于从模型生成样本，避免梯度计算（通过 @torch.no_grad()）。
    
    参数:
    - model: 模型实例，用于生成均值和方差。
    - x: 输入数据的当前状态。
    - cond: 条件变量，用于控制生成过程。
    - t: 时间步数，表示当前的扩散步骤。
    
    返回:
    - 生成的样本（model_mean + model_std * noise）以及空的 values 张量。
    """
    # 通过模型计算均值和对数方差
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    # 根据对数方差计算标准差
    model_std = torch.exp(0.5 * model_log_variance)

    # 当 t == 0 时不添加噪声
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    # 创建一个零张量作为 values
    values = torch.zeros(len(x), device=x.device)
    # 返回生成的样本（均值加上噪声）和 values
    return model_mean + model_std * noise, values

def sort_by_values(x, values):
    """
    根据 values 对输入 x 排序，降序排列。
    
    参数:
    - x: 输入数据。
    - values: 每个输入的对应值，用于排序。
    
    返回:
    - 排序后的 x 和 values。
    """
    # 根据 values 降序排序索引
    inds = torch.argsort(values, descending=True)
    # 使用排序后的索引重新排列 x 和 values
    x = x[inds]
    values = values[inds]
    return x, values

def make_timesteps(batch_size, i, device):
    """
    创建一个时间步张量，表示当前 batch 中的时间步数。
    
    参数:
    - batch_size: 当前批次的大小。
    - i: 当前的时间步数。
    - device: 指定设备（如 CPU 或 GPU）。
    
    返回:
    - 包含时间步数 i 的张量。
    """
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

# 定义高斯扩散模型类，用于处理数据的扩散和逆扩散过程
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,                # 扩散模型的核心网络
        horizon,              # 时间步数，即序列的长度
        observation_dim,      # 观测数据的维度（位置数据）
        action_dim,           # 动作数据的维度（速度数据）
        n_timesteps=1000,     # 扩散过程的时间步数
        loss_type="l1",       # 损失函数类型（默认为L1损失）
        clip_denoised=False,  # 是否对去噪后的数据进行裁剪
        predict_epsilon=True, # 是否直接预测噪声
        action_weight=1.0,    # 动作损失权重
        loss_discount=1.0,    # 损失折扣因子
        loss_weights=None,    # 自定义损失权重
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim  # 转换维度为观测+动作
        self.model = model

        # 生成 beta 时间表，控制噪声增加的步长
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas  # 计算 alpha (1 - beta)
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # 累积乘积用于生成累计 alpha
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # 前一时间步的累计 alpha

        # 设置扩散模型的参数
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # 注册缓冲区，避免在前向传播时更改
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_{t-1}) 相关计算
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # 后验分布的计算公式 q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## posterior log variance 的计算并进行裁剪，以避免最初扩散链中的 0 值
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        ## 计算损失的权重，并初始化损失函数
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)  # 选择损失函数
        
    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        设置损失权重，用于在计算损失时调整轨迹中的每个时间步和每个维度的权重。

        参数:
        - action_weight : float
            首个动作损失的权重系数，用于增大或减小第一个动作对损失的影响。
        - discount : float
            每个时间步的折扣因子。通过 discount^t 控制各时间步损失的权重逐步衰减。
        - weights_dict : dict
            用于对观测数据中特定维度设置权重的字典。格式为 { i: c }，其中 i 是维度索引，c 是权重系数。
        """
        self.action_weight = action_weight

        # 初始化维度权重为1（没有特定权重），长度为 transition_dim（即观测维度和动作维度的总和）
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # 针对指定的维度设置权重值，如果 weights_dict 为 None，默认为空字典
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w  # 将指定维度的权重调整为字典中指定的值

        # 对轨迹的每个时间步的损失进行折扣，即 discount^t，使后期时间步的损失占比逐渐降低
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()  # 标准化折扣以保持整体权重平衡
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)  # 生成一个 (horizon, transition_dim) 形状的权重矩阵

        # 手动设置第一个动作的权重
        loss_weights[0, : self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        根据噪声预测初始状态 x0。

        参数:
        - x_t: 在时间步 t 的状态。
        - t: 当前的时间步数。
        - noise: 模型预测的噪声。

        返回:
        - 去噪后的 x0 状态。

        注意:
        - 如果 self.predict_epsilon 为 True，模型输出噪声；否则直接预测 x0。
        """
        if self.predict_epsilon:
            # 使用累积 alpha 值和噪声计算 x0
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差，用于逆扩散过程。

        参数:
        - x_start: 初始状态 x0。
        - x_t: 在时间步 t 的状态。
        - t: 当前时间步数。

        返回:
        - 后验均值、后验方差和后验对数方差。
        """
        # 计算后验均值，基于 t 时刻的状态和初始状态
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 提取后验方差和对数方差
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """
        使用模型计算时间步 t 时的均值和方差，用于逆扩散采样。

        参数:
        - x: 在时间步 t 的状态。
        - cond: 条件变量，用于控制生成过程。
        - t: 当前时间步数。

        返回:
        - 模型均值、后验方差和后验对数方差。
        """
        # 使用噪声预测初始状态 x0
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        # 根据配置决定是否裁剪去噪后的输出
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)  # 限制生成数据在 [-1, 1] 区间
        else:
            assert RuntimeError()

        # 基于去噪后的 x0 计算模型均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        verbose=True,
        return_chain=False,
        sample_fn=default_sample_fn,
        conditioning_fn=apply_conditioning,
        starting_motion: torch.Tensor = None,
        max_timesteps: int = None,
        **sample_kwargs
    ):
        """
        逆扩散采样循环，逐步生成采样数据。

        参数:
        - shape: 生成数据的形状。
        - cond: 条件变量。
        - verbose: 是否打印进度。
        - return_chain: 是否返回采样链（采样的所有步骤）。
        - sample_fn: 采样函数，用于生成下一个时间步。
        - conditioning_fn: 条件函数，用于在采样过程中添加条件。
        - starting_motion: 起始运动数据，如果指定则作为采样的起点。
        - max_timesteps: 最大时间步数，如果指定则限制扩散步数。

        返回:
        - 一个包含采样数据、值和采样链的 Sample 对象。
        """
        # 获取设备（CPU 或 GPU）
        device = self.betas.device

        # 设置批次大小
        batch_size = shape[0]
        # 如果提供了起始运动数据，则使用该数据作为 x 的起点，否则随机初始化 x
        if starting_motion is not None:
            x = starting_motion.to(device)
        else:
            x = torch.randn(shape, device=device)
        x = conditioning_fn(x)  # 对初始 x 应用条件函数

        # 如果 return_chain 为 True，初始化链以存储每一步的采样结果
        chain = [x] if return_chain else None

        # 确定最大时间步数，如果未指定则使用 n_timesteps
        timesteps = self.n_timesteps if max_timesteps is None else max_timesteps
        progress = utils.Progress(timesteps) if verbose else utils.Silent()  # 显示进度

        # 逆序遍历时间步，从大到小逐步采样
        for i in reversed(range(0, timesteps)):
            t = make_timesteps(batch_size, i, device)  # 当前时间步的张量
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)  # 采样下一个状态
            x = conditioning_fn(x)  # 应用条件函数

            # 更新进度条，显示当前时间步和值的范围
            progress.update(
                {"t": i, "vmin": values.min().item(), "vmax": values.max().item()}
            )
            if return_chain:
                chain.append(x)  # 将采样结果添加到链中

        # 完成采样后，记录最终时间点
        progress.stamp()

        # 根据值对最终采样的结果排序
        x, values = sort_by_values(x, values)
        if return_chain:
            chain = torch.stack(chain, dim=1)  # 将链转换为张量形式
        return Sample(x, values, chain)  # 返回包含采样结果、值和采样链的 Sample 对象
    
    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
        在给定条件下生成样本的函数。该函数用于生成具有特定条件（如初始状态或时间步）的样本序列。

        参数:
        - cond: 条件变量，通常为一个列表，包含 [(time, state), ...] 格式的条件信息。
        - horizon: 时间步长，可选。如果未指定，则使用默认的 self.horizon。
        - **sample_kwargs: 其他采样参数。

        返回:
        - 通过逆扩散过程生成的样本。
        """
        device = self.betas.device  # 获取模型的设备（如 GPU 或 CPU）
        batch_size = len(cond[0])  # 获取批次大小，即条件的第一项的长度
        horizon = horizon or self.horizon  # 如果未指定 horizon，则使用默认的时间步数
        shape = (batch_size, horizon, self.transition_dim)  # 设定生成数据的形状

        # 调用 p_sample_loop 执行逆扩散过程并返回生成的样本
        return self.p_sample_loop(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        用于在扩散过程中的时间步 t 添加噪声生成样本的函数，即向样本中加入噪声。

        参数:
        - x_start: 初始无噪声样本，即 x0。
        - t: 当前时间步数。
        - noise: 噪声张量，如果为 None，则随机生成。

        返回:
        - 加噪后的样本 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x_start)  # 生成与 x_start 形状相同的标准正态分布噪声

        # 使用时间步 t 的累计 alpha 值对初始样本和噪声进行加权，生成 x_t
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample  # 返回在时间步 t 添加噪声后的样本

    def p_losses(self, x_start, cond, t):
        """
        计算给定时间步 t 的损失函数，用于训练扩散模型。

        参数:
        - x_start: 初始无噪声样本 x0。
        - cond: 条件变量，包含用于生成过程中的条件信息。
        - t: 当前时间步数。

        返回:
        - 损失值和损失信息（info）。
        """
        noise = torch.randn_like(x_start)  # 生成与 x_start 形状相同的标准正态噪声

        # 使用 q_sample 在时间步 t 向 x_start 添加噪声，生成 x_t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy)  # 应用条件，将样本调整为符合条件的形式

        # 模型通过 x_noisy 和 cond 重构噪声
        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon)  # 应用条件调整模型输出

        assert noise.shape == x_recon.shape  # 确保重构后的样本形状与噪声形状一致

        # 计算损失：如果预测 epsilon，则与噪声对比，否则与初始无噪声样本对比
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info  # 返回损失和损失的详细信息

    def loss(self, x, *args):
        """
        计算总体损失的函数，基于当前批次中的时间步数随机抽样。

        参数:
        - x: 输入样本数据。
        - *args: 其他参数（如条件）。

        返回:
        - 计算的损失值和损失信息。
        """
        batch_size = len(x)  # 获取当前批次大小
        # 从时间步数中随机抽取一个时间步，用于损失计算
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)  # 调用 p_losses 计算损失

    def forward(self, cond, *args, **kwargs):
        """
        前向传播函数，用于生成条件采样的样本。

        参数:
        - cond: 条件变量。
        - *args, **kwargs: 其他参数。

        返回:
        - 在给定条件下生成的样本。
        """
        return self.conditional_sample(cond, *args, **kwargs)  # 执行条件采样

# 定义 ValueDiffusion 类，它是 GaussianDiffusion 的子类，专注于值函数的扩散
class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        """
        用于训练 ValueDiffusion 模型的损失计算方法，支持目标值作为对比。

        参数:
        - x_start: 初始无噪声样本。
        - cond: 条件变量。
        - target: 目标值，用于损失计算的对比。
        - t: 当前时间步数。

        返回:
        - 损失值和损失信息。
        """
        noise = torch.randn_like(x_start)  # 生成噪声

        # 添加噪声生成 x_t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 使用 apply_conditioning 将条件 cond 应用于 x_noisy 的前 action_dim 个维度
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        # 使用模型预测
        pred = self.model(x_noisy, cond, t)

        # 计算损失，预测值与目标值进行对比
        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        """
        前向传播函数，用于在 ValueDiffusion 中通过模型进行生成。
        
        参数:
        - x: 输入样本。
        - cond: 条件变量。
        - t: 当前时间步数。
        
        返回:
        - 模型的预测结果。
        """
        return self.model(x, cond, t)  # 使用模型进行预测
