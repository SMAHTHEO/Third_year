import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from tqdm import tqdm

from Unets import ConditionalUnet  # 导入条件U-Net模型
from utils import ddpm_schedules, load_MNIST  # 导入调度函数和数据加载函数

# 定义训练函数
def train_diffusion(diffusion, device, n_epoch=50, sample_dir='log/classifier_free'):
    diffusion.to(device)  # 将模型移动到指定设备

    dataloader = load_MNIST()  # 加载MNIST数据集
    optim = torch.optim.Adam(diffusion.parameters(), lr=2e-4)  # 定义Adam优化器

    # 如果保存生成样本的目录不存在，则创建
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # 开始训练循环
    for i in range(n_epoch):
        diffusion.train()  # 设置模型为训练模式

        pbar = tqdm(dataloader)  # 使用tqdm显示进度条
        loss_ema = None  # 初始化损失的指数移动平均值
        for x, y in pbar:
            optim.zero_grad()  # 清零梯度
            x = x.to(device)  # 将数据移动到设备
            y = y.to(device)
            loss = diffusion(x, y)  # 前向传播，计算损失
            loss.backward()  # 反向传播，计算梯度

            # 计算损失的指数移动平均值，便于观察训练过程中的损失变化
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")  # 更新进度条的描述
            optim.step()  # 更新模型参数

        diffusion.eval()  # 设置模型为评估模式
        with torch.no_grad():
            # 生成样本，用于观察模型的生成效果
            y = (torch.arange(0, 40) % 10).to(device)  # 创建条件标签，包含0-9的数字，每个数字4次，共40个样本
            xh = diffusion.sample(40, y, (1, 28, 28))  # 生成40张28x28的图像
            grid = make_grid(xh, nrow=10)  # 将生成的图像排列成网格

            # 保存生成的图像
            save_image(grid, f"{sample_dir}/ddpm_sample_{i}.png")

        # 保存模型的状态字典，便于后续加载或继续训练
        torch.save(diffusion.state_dict(), f"{sample_dir}/ddpm_mnist_{i}.pth")

# 定义无分类器指导的DDPM模型类
class ClassifierFreeDDPM(nn.Module):
    def __init__(self, eps_model, betas, T, device, drop_prob=0.1):
        super(ClassifierFreeDDPM, self).__init__()
        self.eps_model = eps_model  # 噪声预测模型（条件U-Net）

        # 注册扩散过程中的各种参数（α、β、σ等）
        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)  # 将参数注册为buffer，模型保存时会保存这些参数

        self.T = T  # 总的时间步数
        self.mse_loss = nn.MSELoss()  # 定义均方误差损失函数
        self.device = device  # 设备（CPU或GPU）
        self.drop_prob = drop_prob  # 丢弃条件信息的概率，用于无分类器的指导

    def forward(self, x, y):
        """
        前向传播，用于训练模型。
        输入：
            x: 原始图像数据
            y: 图像对应的标签（条件信息）
        返回：
            loss: 预测噪声与真实噪声之间的MSE损失
        """
        # 随机采样时间步t，范围从1到T
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)

        # 生成与x形状相同的标准正态分布噪声eps
        eps = torch.randn_like(x)

        # 根据前向扩散公式，生成添加噪声后的图像x_t
        # x_t = sqrt(α_bar_t) * x + sqrt(1 - α_bar_t) * eps
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + \
              torch.sqrt(1 - self.alpha_bar[t, None, None, None]) * eps

        # 随机决定是否使用条件信息（以drop_prob的概率丢弃条件）
        # context_mask为0表示使用条件信息，为1表示丢弃条件信息
        context_mask = torch.bernoulli(torch.zeros_like(y) + self.drop_prob).to(self.device)

        # 使用噪声预测模型预测噪声，并计算与真实噪声的MSE损失
        # 将时间步t归一化到[0,1]范围内，作为模型的输入
        return self.mse_loss(eps, self.eps_model(x_t, y, t / self.T, context_mask))

    def sample(self, n_sample, y, size, guide_w=0.0):
        """
        从噪声中采样生成条件图像，使用无分类器的自由指导。
        输入：
            n_sample: 生成的样本数量
            y: 条件标签（如数字类别）
            size: 图像的尺寸（通道数，高度，宽度）
            guide_w: 指导的强度，w > 0 表示更强的指导
        返回：
            x_t: 生成的图像
        """
        # 我们遵循“Classifier-Free Diffusion Guidance”论文中的指导采样方案
        # 为了提高前向传播的效率，我们将有条件和无条件的版本合并到一个批次中
        # 然后根据指导权重guide_w调整输出

        # 初始化为标准正态分布的噪声图像
        x_t = torch.randn(n_sample, *size).to(self.device)

        # 扩展批次，将批次大小加倍，以包含有条件和无条件的样本
        y_double = y.repeat(2)  # 重复y，形成两倍大小的y
        context_mask = torch.zeros_like(y_double).to(self.device)  # 初始化context_mask为0（使用条件）
        context_mask[n_sample:] = 1.0  # 将后半部分的context_mask设置为1（丢弃条件）

        # 逆向扩散过程，从T到1
        for t in reversed(range(self.T)):
            # 如果t > 1，则采样标准正态噪声z，否则z为0
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0

            # 将时间步t归一化，并调整形状以匹配模型输入
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # 扩展批次，包含有条件和无条件的样本
            x_t_double = x_t.repeat(2, 1, 1, 1)  # 形状变为 (2 * n_sample, ...)
            t_is_double = t_is.repeat(2, 1, 1, 1)  # 时间步也要重复

            # 使用噪声预测模型预测噪声eps
            eps = self.eps_model(x_t_double, y_double, t_is_double, context_mask)
            eps1 = eps[:n_sample]      # 有条件的预测
            eps2 = eps[n_sample:]      # 无条件的预测

            # 根据指导权重guide_w调整预测的噪声
            # eps = (1 + w) * eps_cond - w * eps_uncond
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            # 根据逆扩散公式更新x_t
            x_t = 1 / torch.sqrt(self.alpha[t]) * (x_t - eps * (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])) + \
                  self.sigma[t] * z

        return x_t  # 返回生成的图像

# 主函数
if __name__ == "__main__":
    # 检查可用的设备（CUDA、MPS或CPU）
    if torch.cuda.is_available():  # 如果有NVIDIA的GPU可用
        device_type = "cuda"
    elif torch.backends.mps.is_available():  # 如果有Apple Silicon的GPU可用
        device_type = "mps"
    else:
        device_type = "cpu"

    device = torch.device(device_type)  # 选择最佳可用设备

    # 实例化无分类器指导的DDPM模型
    cfd = ClassifierFreeDDPM(
        eps_model=ConditionalUnet(in_channels=1),  # 噪声预测模型，输入通道为1（灰度图像）
        betas=(1e-4, 0.02),  # β的范围
        T=1000,  # 时间步数
        device=device,  # 设备
        drop_prob=0.1  # 丢弃条件信息的概率
    )

    # 开始训练扩散模型
    train_diffusion(diffusion=cfd, device=device)
