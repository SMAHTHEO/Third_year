import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

from tqdm import tqdm

from Unets import ConditionalUnet  # 导入条件U-Net模型
from utils import ddpm_schedules, load_MNIST  # 导入调度函数和数据加载函数

from Unets import Unet
from ddpm import DDPM  # 导入基本的DDPM模型

# 定义用于计算分类器梯度的函数
def classifier_grad_fn(x, classifier, y, _scale=1):
    """
    返回分类器关于输入x的梯度，即 d_log(p(y|x)) / dx
    """
    assert y is not None  # 确保标签y不为空
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)  # 使x可求导
        logits = classifier(x_in)  # 获取分类器的输出（未经过Softmax）
        log_probs = F.log_softmax(logits, dim=-1)  # 计算对数概率
        selected = log_probs[range(len(logits)), y.view(-1)]  # 选择目标类别的对数概率
        # 计算关于x_in的梯度，并乘以缩放因子_scale
        grad = torch.autograd.grad(selected.sum(), x_in)[0] * _scale
        return grad  # 返回梯度

# 定义带分类器指导的DDPM模型类
class GuidedDDPM(nn.Module):
    def __init__(self, eps_model, betas, T, device, classifier, _grad_fn, _scale=1.0):
        super(GuidedDDPM, self).__init__()
        self.eps_model = eps_model  # 噪声预测模型

        # 注册扩散过程中的各种参数（α、β、σ等）
        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)  # 将参数注册为buffer

        self.T = T  # 总的时间步数
        self.mse_loss = nn.MSELoss()  # 定义均方误差损失函数
        self.device = device  # 设备（CPU或GPU）

        # 分类器指导相关
        self.classifier = classifier  # 预训练的分类器模型
        self._grad_fn = _grad_fn  # 用于计算梯度的函数
        self._scale = _scale  # 指导强度的缩放因子

    def forward(self, x):
        """
        前向传播，用于训练模型。
        输入：
            x: 原始图像数据
        返回：
            loss: 预测噪声与真实噪声之间的MSE损失
        """
        # 随机采样时间步t，范围从1到T
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)

        # 生成与x形状相同的标准正态分布噪声eps
        eps = torch.randn_like(x)

        # 根据前向扩散公式，生成添加噪声后的图像x_t
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + \
              torch.sqrt(1 - self.alpha_bar[t, None, None, None]) * eps

        # 使用噪声预测模型预测噪声，并计算与真实噪声的MSE损失
        return self.mse_loss(eps, self.eps_model(x_t, t / self.T))

    def sample(self, n_sample, y, size):
        """
        从噪声中采样生成条件图像，使用分类器指导。
        输入：
            n_sample: 生成的样本数量
            y: 目标类别标签
            size: 图像的尺寸（通道数，高度，宽度）
        返回：
            x_t: 生成的图像
        """
        # 初始化为标准正态分布的噪声图像
        x_t = torch.randn(n_sample, *size).to(self.device)

        # 逆向扩散过程，从T到1
        for t in reversed(range(self.T)):
            # 如果t > 1，则采样标准正态噪声z，否则z为0
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0

            # 将时间步t归一化，并调整形状以匹配模型输入
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # 使用噪声预测模型预测噪声eps
            eps = self.eps_model(x_t, t_is)

            # 分类器指导部分
            # 计算x_t的预测均值
            x_t_mean = 1 / torch.sqrt(self.alpha[t]) * (x_t - eps * (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t]))
            x_t_variance = self.sigma[t]  # 计算x_t的方差

            # 计算分类器关于x_t_mean的梯度，用于指导生成过程
            gradient = self._grad_fn(x_t_mean, self.classifier, y, self._scale)

            # 将梯度乘以方差，调整x_t_mean，实现指导
            x_t_mean = x_t_mean + x_t_variance * gradient

            # 更新x_t，准备进入下一个时间步
            x_t = x_t_mean + self.sigma[t] * z

        return x_t  # 返回生成的图像

# 定义训练分类器的函数
def train_classifier(classifier, diffusion, device, n_epoch=10):
    print('########## Train classifier ##########')
    classifier.to(device)  # 将分类器移动到指定设备

    dataloader = load_MNIST()  # 加载MNIST数据集
    optim = torch.optim.Adam(classifier.parameters(), lr=2e-4)  # 定义Adam优化器
    ce_loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    def pred_accuracy(preds, labels):
        y_pred = torch.max(preds, 1)[1]  # 获取预测的类别
        return torch.mean(y_pred.eq(labels).float())  # 计算准确率

    save_interval = 5  # 设置模型保存间隔

    classifier_dir = 'log/classifier'  # 定义分类器模型保存的目录
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)

    # 开始训练循环
    for i in range(n_epoch):
        classifier.train()  # 设置分类器为训练模式

        pbar = tqdm(dataloader)  # 使用tqdm显示进度条
        for x, y in pbar:
            optim.zero_grad()  # 清零梯度
            x = x.to(device)
            y = y.to(device)

            # 随机采样时间步t，范围从1到T
            t = torch.randint(1, diffusion.T, (x.shape[0],)).to(device)
            eps = torch.randn_like(x)  # 生成噪声
            # 对图像x添加噪声，得到x_t
            x_t = torch.sqrt(diffusion.alpha_bar[t, None, None, None]) * x + \
                  torch.sqrt(1 - diffusion.alpha_bar[t, None, None, None]) * eps

            preds = classifier(x_t)  # 使用分类器预测类别

            loss = ce_loss(preds, y)  # 计算交叉熵损失
            loss.backward()  # 反向传播，计算梯度
            acc = pred_accuracy(preds, y)  # 计算准确率

            pbar.set_description(f"loss: {loss:.4f}, acc: {acc:.3f}")  # 更新进度条的描述
            optim.step()  # 更新分类器参数

        # 保存分类器模型
        if (i + 1) % save_interval == 0:
            torch.save(classifier.state_dict(), f"{classifier_dir}/classifier_mnist_{i}.pth")

# 定义指导采样的函数
def guided_sampling(device):
    # 加载预训练的DDPM模型
    ddpm = DDPM(eps_model=Unet(in_channels=1), betas=(1e-4, 0.02), T=1000, device=device)
    ddpm_ckpt_path = 'log/samples/ddpm_mnist_45.pth'  # 模型的检查点路径
    ddpm.load_state_dict(torch.load(ddpm_ckpt_path))  # 加载模型参数
    eps_model = ddpm.eps_model  # 获取噪声预测模型
    eps_model.to(device)

    # 构建分类器模型
    from Unets import EncoderUnet
    classifier = EncoderUnet(in_channels=1)
    classifier.to(device)

    # 实例化带分类器指导的DDPM模型
    g_ddpm = GuidedDDPM(
        eps_model=eps_model,
        betas=(1e-4, 0.02),
        T=1000,
        device=device,
        classifier=classifier,
        _grad_fn=classifier_grad_fn,
        _scale=1.0  # 指导强度的缩放因子
    )
    g_ddpm.to(device)

    # 训练分类器
    train_classifier(classifier, g_ddpm, device)

    # 如果有已保存的分类器模型，可以加载
    # classifier_ckpt_path = 'log/classifier/classifier_mnist_9.pth'
    # classifier.load_state_dict(torch.load(classifier_ckpt_path))

    sample_dir = 'log/samples_guided'  # 定义生成样本保存的目录
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    g_ddpm.eval()  # 设置模型为评估模式
    with torch.no_grad():
        y = (torch.arange(0, 40) % 10).to(device)  # 创建条件标签，包含0-9的数字，每个数字4次，共40个样本
        xh = g_ddpm.sample(40, y, (1, 28, 28))  # 生成40张28x28的图像
        grid = make_grid(xh, nrow=10)  # 将生成的图像排列成网格

        # 保存生成的图像
        save_image(grid, f"{sample_dir}/guided_sample.png")

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
    guided_sampling(device)  # 开始指导采样过程
