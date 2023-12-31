from .nn import init_weights, init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i][0]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        # self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,
        #                        posterior=self.posterior)
        self.conv_layer = nn.Conv2d(int(128), 2 * int(self.latent_dim[0]), (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = input #elf.encoder(input)
        self.show_enc = encoding

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=1, keepdim=True)
        encoding = torch.mean(encoding, dim=2, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=1)

        mu = mu_log_sigma[:self.latent_dim[0]]
        log_sigma = mu_log_sigma[self.latent_dim[0]:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


if __name__ == '__main__':
    # 导入必要的库
    import torch

    # 设定输入图像的通道数、滤波器数量、卷积块中的卷积层数、潜在维度等参数
    input_channels = 3  # 假设输入图像有3个通道
    num_filters = [32, 64, 128]  # 3个卷积块中的滤波器数量
    no_convs_per_block = 2  # 每个卷积块中的卷积层数
    latent_dim = 16  # 潜在维度

    # 创建一个示例输入图像（这里使用随机生成的图像，你可以替换为实际图像数据）
    sample_input = torch.randn(1, input_channels, 128, 128)  # 假设图像大小是128x128

    # 创建一个示例分割掩码（如果需要，你可以提供分割掩码，否则可以将其设置为None）
    sample_segm = None

    # 创建AxisAlignedConvGaussian模型
    posterior_model = AxisAlignedConvGaussian(input_channels, num_filters, no_convs_per_block, latent_dim, None,
                                              posterior=True)

    # 将模型设置为评估模式
    posterior_model.eval()

    # 前向传播，获取输出分布
    output_distribution = posterior_model(sample_input, sample_segm)

    # 打印输出分布的一些属性
    print("Mean (mu):", output_distribution.mean)
    print("Log Standard Deviation (log_sigma):", output_distribution.stddev)
