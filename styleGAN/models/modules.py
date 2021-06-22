'''
    https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py
    https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
'''

import torch
from torch import nn
import torch.nn.functional as F


###         UNIVERSAL           ###
def get_activation(name='lrelu', inplace=True):
    '''
        Get the module for a specific activation function and its gain
    '''
    if name == 'relu':
        return nn.ReLU(inplace=inplace), 2**0.5
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=inplace), 2**0.5
    elif name == 'sigmoid':
        return nn.Sigmoid(), 1
    elif name in [None, 'linear']:
        return nn.Identity(), 1
    else:
        raise ValueError('Activation "{}" not available.'.format(name))


class PixelNormLayer(nn.Module):
    '''
        In this case, it is just for the input Z;
    '''
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, dim=1):
        return x * torch.rsqrt(torch.mean(x**2, dim=dim, keepdim=True) + self.epsilon)


class FCLayer(nn.Module):
    """
        Linear layer with equalized learning rate and custom learning rate multiplier.
    """
    def __init__(self, input_size, output_size, activation=None, use_wscale=False, lrmul=1, bias=0):
        super(FCLayer, self).__init__()
        self.act, gain = get_activation(activation)
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias == 0:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        elif bias == 1:
            self.bias = torch.nn.Parameter(torch.ones(output_size))
            self.b_mul = lrmul            
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return self.act(F.linear(x, self.weight * self.w_mul, bias))


class NoiseLayer(nn.Module):
    """
        Add noise. 
        noise is per pixel (constant over channels) with per-channel weight
    """
    def __init__(self, channels):
        super(NoiseLayer, self).__init__()
        # learned per-channel scaling factor to noise input
        self.weight = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class styleMod(nn.Module):
    """
        Modulate the weight
        'use_mean'=False for styleGAN2; 'use_mean'=True for styleGAN1;
    """
    def __init__(self, latent_size, channels, activation='linear', use_wscale=False, lrmul=1, bias=1, use_mean=False):
        super(styleMod, self).__init__()
        if use_mean:
            self.lin = FCLayer(latent_size, channels*2, activation=activation, use_wscale=use_wscale, lrmul=lrmul, bias=bias)
        else:
            self.lin = FCLayer(latent_size, channels, activation=activation, use_wscale=use_wscale, lrmul=lrmul, bias=bias)
        self.use_mean = use_mean

    def forward(self, x, latent):
        style = self.lin(latent)
        if self.use_mean:
            shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
            style = style.view(shape)
            x = x * (style[:, 0] + 1.) + style[:, 1]
        else:
            # For styleGAN2: In this case, x is the weight.
            shape = [style.size(0), 1, -1] + (x.dim() - 2) * [1]
            style = style.view(shape)
            x = x.unsqueeze(0)
            x = x * style
        return x


class BlurLayer(nn.Module):
    '''
        To mitigate the blocky results, the generator blurs the layer, by convolving with the simplest possible smoothing kernel.
    '''
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is not None:
            kernel = kernel
            kernel = torch.tensor(kernel, dtype=torch.float32)
            kernel = kernel[:, None] * kernel[None, :]
            kernel = kernel[None, None]
            if normalize:
                kernel = kernel / kernel.sum()
            if flip:
                kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride
    
    def forward(self, x):
        self.kernel = self.kernel.to(x.device)
        if self.kernel is not None:
            # expand kernel channels
            kernel = self.kernel.expand(x.size(1), -1, -1, -1)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.kernel.size(2)-1)/2),
                groups=x.size(1),
            )
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, truncation_cutoff=8, truncation_psi=0.7, beta=0.995):
        super().__init__()
        self.truncation_cutoff = truncation_cutoff
        self.truncation_psi = truncation_psi
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update_avg(self, last_avg):
        last_avg = last_avg.mean(0) if last_avg.dim() == 2 else last_avg
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.truncation_psi)
        do_trunc = (torch.arange(x.size(1)) < self.truncation_cutoff).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = (y**2).mean(dim=0)              # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


###          ONLY FOR STYLEGAN2            ###
class weightDemod(nn.Module):
    '''
        Demodulate the weight.
    '''
    def __init__(self, epsilon=1e-8):
        super(weightDemod, self).__init__()
        self.epsilon = epsilon

    def forward(self, w):
        return w * torch.rsqrt((w**2).sum(dim=[2,3,4], keepdim=True) + self.epsilon)


class ConvLayer(nn.Module):
    '''
        Convoluational layer equipped with equalized learning rate, custom learning rate multiplier, 
                                           modulated layer and demodulated layer.
    '''
    def __init__(self, input_channels, output_channels, latent_size=0, kernel_size=3, stride=1, activation='linear', use_wscale=True, lrmul=1, bias=False, mod=True, demod=True):
        super(ConvLayer, self).__init__()
        self.act, gain = get_activation(activation)
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5) # He init
        
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        if mod:
            self.mod = styleMod(latent_size, input_channels, activation='linear', use_wscale=use_wscale, lrmul=lrmul, bias=1, use_mean=False)
        else:
            self.mod = None

        self.demod = weightDemod() if demod else None

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, latent=None):
        weight = self.weight
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        if self.mod is not None:
            weight = self.mod(weight, latent)
        if self.demod is not None:
            weight = self.demod(weight)

        if latent is not None:
            weight = weight.view(-1, *weight.shape[2:])
            bs = x.size(0)
            x = x.view(1, -1, *x.shape[2:])
            out = F.conv2d(x, weight * self.w_mul, stride=self.stride, padding=self.kernel_size//2, groups=bs)
            out = out.view(bs, -1, out.size(2), out.size(3))
        else:       # common convoluational layer
            out = F.conv2d(x, weight * self.w_mul, stride=self.stride, padding=self.kernel_size//2,)
        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)
        return self.act(out)


class GSynthesisLayer(nn.Module):
    '''
        Include upsample(optional), conv, niose and bias
    '''
    def __init__(self, input_channels, output_channels, latent_size, kernel_size=3, activation='lrelu', use_wscale=True, lrmul=1, bias=False, mod=True, demod=True, upsample=False):
        super(GSynthesisLayer, self).__init__()
        self.act, _ = get_activation(activation)
        self.conv = ConvLayer(input_channels, output_channels, latent_size, kernel_size, 1, 'linear', use_wscale, lrmul, bias, mod, demod)
        # if upsample:
        #     self.upLayer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.addNoise = NoiseLayer(output_channels)

        self.upsample = upsample

    def forward(self, x, latent):
        if self.upsample:
            # x = self.upLayer(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv(x, latent)
        out = self.addNoise(out)
        out = out + self.bias.view(1, -1, 1, 1)
        
        return self.act(out)


class GSynthesisBlock(nn.Module):
    def __init__(self, 
                in_channels,            # Number of input channels, 0 = first block.
                out_channels,           # Number of output channels.
                img_channels,           # Number of output color channels.
                resolution,             # Resolution of this block.
                latent_size,            # Intermediate latent (W) dimensionality.
                kernel_size=3,            # Size of the convolutional kernel. 
                activation='lrelu', use_wscale=True, lrmul=1, bias=False, mod=True, demod=True,
                arch='skip',            # Architecture: 'orig', 'skip', 'resnet'.
                is_last=False,          # Is this the last block?
                resample_filter=[1,2,1] # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()
        self.num_conv = 0
        self.num_torgb = 0
        self.in_channels = in_channels
        self.is_last = is_last
        self.arch = arch
        self.resample_filter = resample_filter

        if in_channels == 0:
            self.const_input = nn.Parameter(torch.randn(out_channels, resolution, resolution))
        else:
            self.conv0 = GSynthesisLayer(in_channels, out_channels, latent_size, kernel_size, activation, use_wscale, lrmul, bias, mod, demod, upsample=True)
            self.num_conv += 1

        self.conv1 = GSynthesisLayer(out_channels, out_channels, latent_size, kernel_size, activation, use_wscale, lrmul, bias, mod, demod, upsample=False)
        self.num_conv += 1

        if is_last or arch=='skip':
            self.torgb = ConvLayer(out_channels, img_channels, latent_size, kernel_size=1, bias=True, demod=False)
            self.num_torgb += 1

        if in_channels !=0 and arch=='resnet':
            # Implementation of "Up" operation in "skip" architecture is common nn.Upsample, while transposed conv in "resnet" architecture.
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, bias=False),
                BlurLayer(resample_filter),
            )

    def forward(self, x, ws, img=None):
        assert ws.size(1) == (self.num_conv+self.num_torgb)
        w_iter = iter(ws.unbind(dim=1))
        if self.in_channels == 0:
            x = self.const_input
            x = x.unsqueeze(0).repeat([ws.size(0), 1, 1, 1])
        
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter))
        elif self.arch == 'resnet':
            y = self.up(x)
            x = self.conv0(x, next(w_iter))
            x = self.conv1(x, next(w_iter))
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter))
            x = self.conv1(x, next(w_iter))
        
        # toRGB
        # 'img is not None' means self.arch == 'skip'
        if img is not None:
            img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            img = BlurLayer(kernel=self.resample_filter)(img)

        if self.is_last or self.arch == 'skip':
            y = self.torgb(x, next(w_iter))
            img = img.add_(y) if img is not None else y

        return x, img


class DiscriminatorBlock(nn.Module):
    def __init__(self, 
                in_channels,            # Number of input channels, 0 = first block.
                tmp_channels,           # Number of intermediate channels.
                out_channels,           # Number of output channels.
                img_channels,           # Number of output color channels.
                arch='resnet',            # Architecture: 'orig', 'skip', 'resnet'.
                activation='lrelu', use_wscale=True, lrmul=1, bias=True,
                resample_filter=[1,2,1] # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()
        if in_channels == 0 or arch == 'skip':
            self.fromrgb = ConvLayer(img_channels, tmp_channels, stride=1, bias=True, mod=False, demod=False, activation=activation)
        self.conv0 = ConvLayer(tmp_channels, tmp_channels, activation=activation, use_wscale=use_wscale, lrmul=lrmul, bias=bias, mod=False, demod=False)
        self.conv1 = ConvLayer(tmp_channels, out_channels, activation=activation, use_wscale=use_wscale, lrmul=lrmul, bias=bias, mod=False, demod=False)
        if arch == 'resnet':
            self.down = nn.Sequential(
                nn.Conv2d(tmp_channels, out_channels, 3, 1, padding=1),
                BlurLayer(resample_filter, stride=2),
            )
        
        self.in_channels = in_channels
        self.arch = arch
        self.resample_filter = resample_filter

    def forward(self, x=None, img=None):
        if self.in_channels == 0 or self.arch == 'skip':
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            if self.arch == 'skip':
                # downsample in 'skip' architecture
                img = BlurLayer(self.resample_filter, stride=2)

        if self.arch == 'resnet':
            y = self.down(x)
            x = self.conv0(x)
            x = self.conv1(x)
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        return x, img


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self, 
                in_channels,                    # Number of input channels.
                resolution,                     # Resolution of this block.
                img_channels,                   # Number of input color channels.
                arch                = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
                mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
                mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
                activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
    ):
        super(DiscriminatorEpilogue, self).__init__()
        if arch == 'skip':
            self.fromrgb = ConvLayer(img_channels, in_channels, stride=1, bias=True, mod=False, demod=False, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = ConvLayer(in_channels+mbstd_num_channels, in_channels, activation=activation, mod=False, demod=False)
        self.fc = FCLayer(in_channels*(resolution**2), in_channels, activation=activation, use_wscale=True)
        self.out = FCLayer(in_channels, 1, use_wscale=True)

        self.arch = arch

    def forward(self, x=None, img=None):
        if self.arch == 'skip':
            x = x + self.fromrgb(img)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        return x
        
