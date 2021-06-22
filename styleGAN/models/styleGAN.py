'''
    code is mainly adopted from:
    1. https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
'''

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class FCLayer(nn.Module):
    """
        Linear layer with equalized learning rate and custom learning rate multiplier.
    """
    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super(FCLayer, self).__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class NoiseLayer(nn.Module):
    """
        adds noise. noise is per pixel (constant over channels) with per-channel weight
    """
    def __init__(self, channels):
        super().__init__()
        # learned per-channel scaling factor to noise input
        self.weight = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class StyleMod(nn.Module):
    '''
        apply learned style to each feature map
    '''
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = FCLayer(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)
        
    def forward(self, x, latent):
        # import ipdb; ipdb.set_trace()
        style = self.lin(latent) # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
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
        if self.kernel is not None:
            # expand kernel channels
            kernel = self.kernel.expand(x.size(1), -1, -1, -1)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.kernel.size(2)-1)/2),
                groups=x.size(1)
            )
        return x
        
# Upsample method 1. The another one is transposed Conv.
class Upscale2d(nn.Module):
    '''
        Set a block of 2x2 pixels to the value of the pixel to arrive an image that is scaled by 2. 
        No fancy stuff like bilinear or bicubic interpolation. 
    '''
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        assert x.dim() == 4
        if self.gain != 1:
            x = x * self.gain
        if self.factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class ConvLayer(nn.Module):
    """
        Conv layer with equalized learning rate and custom learning rate multiplier.
        "intermediate" option means apply the blurring layer or not.
    """
    def __init__(self, input_channels, output_channels, kernel_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True, intermediate=None):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5) # He init
        self.kernel_size = kernel_size
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
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
    
        if self.intermediate is None:
            x = F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size//2)
        else:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size//2)
            x = self.intermediate(x)
            if bias is not None:
                x = x + bias.view(1, -1, 1, 1)
        return x


class LayerEpilogue(nn.Module):
    """
        The layer contains B(noise operation), A(style operation) and AdaIN(adaptive instance normalization).
    """

    def __init__(self, channels, dlatent_size, use_wscale, use_noise=True, use_pixel_norm=False, use_instance_norm=True, use_styles=True):
        super().__init__()

        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', nn.LeakyReLU(0.2, inplace=True)))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNormLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))

        self.top_epi = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class MappingNet(nn.Module):
    def __init__(self, z_dim=512, hid_dim=512, lrmul=0.01, gain=2**(0.5), use_wscale=True):
        super(MappingNet, self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ('pixel_norm', PixelNormLayer()),
            ('dense0', FCLayer(z_dim, hid_dim, gain=gain, lrmul=lrmul, use_wscale=use_wscale)),
            ('dense0_act', nn.LeakyReLU(0.2, True)),
            ('dense1', FCLayer(hid_dim, hid_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', nn.LeakyReLU(0.2, True)),
            ('dense2', FCLayer(hid_dim, hid_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', nn.LeakyReLU(0.2, True)),
            ('dense3', FCLayer(hid_dim, hid_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', nn.LeakyReLU(0.2, True)),
            ('dense4', FCLayer(hid_dim, hid_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', nn.LeakyReLU(0.2, True)),
            ('dense5', FCLayer(hid_dim, hid_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', nn.LeakyReLU(0.2, True)),
            ('dense6', FCLayer(hid_dim, hid_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', nn.LeakyReLU(0.2, True)),
            ('dense7', FCLayer(hid_dim, z_dim, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', nn.LeakyReLU(0.2, True))            
        ]))

    def forward(self, x):
        x = self.layer(x)
        return x

class InputBlock(nn.Module):
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = FCLayer(dlatent_size, nf*16, gain=gain/4, use_wscale=use_wscale) # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles)
        self.conv = ConvLayer(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles)
        
    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x

class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, res_exp):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        if res_exp < 7:
            self.up_sample = Upscale2d()
        else:
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv0 = ConvLayer(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale, intermediate=blur)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles)
        self.conv1 = ConvLayer(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles)
            
    def forward(self, x, dlatents_in_range):
        x = self.up_sample(x)
        x = self.conv0(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x

class G_synthesis(nn.Module):
    def __init__(self,
        dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
        num_channels        = 3,            # Number of output color channels.
        resolution          = 1024,         # Output resolution.
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        use_styles          = True,         # Enable style inputs?
        const_input_layer   = True,         # First layer is a learned constant?
        use_noise           = True,         # Enable noise inputs?
        use_wscale          = True,         # Enable equalized learning rate?
        use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
        use_instance_norm   = True,         # Enable instance normalization?
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        gain                = 2**(0.5),
        ):        
        super().__init__()

        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        # as this example, we get num_layers = 18.
        self.num_layers = (resolution_log2 - 1) * 2 

        n_featuremap = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = n_featuremap(res-1)
            name = '{s}x{s}'.format(s=2**res)
            if res == 2:
                blocks.append((name, InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                                use_noise, use_pixel_norm, use_instance_norm, use_styles)))
            else:
                blocks.append((name, GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, 
                                                     use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, res)))
            last_channels = channels
        self.torgb = ConvLayer(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))
        
    def forward(self, dlatents_in):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        batch_size = dlatents_in.size(0)       
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2*i:2*i+2])
            else:
                x = m(x, dlatents_in[:, 2*i:2*i+2])
        rgb = self.torgb(x)
        return rgb


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


class styleGenerator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, img_resolution=1024, img_channels=3, truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995, style_mixing_prob=0.9, **kwargs):
        super(styleGenerator, self).__init__()
        self.mapping = MappingNet()
        self.synthesis = G_synthesis()
        if truncation_psi:
            self.truncation = Truncation(torch.zeros(w_dim), truncation_cutoff, truncation_psi, dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, z_latents):
        w_latents = self.mapping(z_latents)
        # expand the w from [bs, dim] to [bs, num_layer, dim], where num_layer means the number of layers in the following synthesis network
        w_latents = w_latents.unsqueeze(1)
        w_latents = w_latents.expand(-1, int(self.synthesis.num_layers), -1)
        # TODO: Apply style mixing regularization and truncation trick
        if self.training and self.truncation is not None:
            self.truncation.update(w_latents[0, 0].detach())
            w_latents = self.truncation(w_latents)
        fake_img = self.synthesis(w_latents)
        return fake_img


##############################################################################################################################################
#   For discriminator 
##############################################################################################################################################

class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gain, use_wscale, blur_filter, res_exp):
        super(DisBlock, self).__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.blur = BlurLayer(blur_filter)
        if res_exp < 7:
            self.conv = ConvLayer(in_channels, out_channels, 3, gain, use_wscale)
            self.down_sample = nn.AvgPool2d(2)
        else:
            self.conv = ConvLayer(in_channels, in_channels, 3, gain, use_wscale)
            self.down_sample = nn.Conv2d(in_channels, out_channels, 2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.blur(x)
        x = self.down_sample(x)
        x = self.act(x)
        return x


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class styleDiscriminator(nn.Module):
    def __init__(self,
                 img_resolution=1024,
                 fmap_base=8192,
                 img_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 gain=2**(0.5),
                 use_wscale=True,
                 blur_filter=None,         # [1, 2, 1] (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.)
    ):
        super(styleDiscriminator, self).__init__()

        self.structure = structure
        self.resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** self.resolution_log2 and img_resolution >= 4
        n_featuremap = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv2d(img_channels, n_featuremap(self.resolution_log2-1), kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        block = []
        for res in range(self.resolution_log2, 2, -1):
            in_channels, out_channels = n_featuremap(res-1), n_featuremap(res-2)
            name = '{s}x{s}'.format(s=2**res)
            block.append((name, DisBlock(in_channels, out_channels, gain, use_wscale, blur_filter, res)))

        self.blocks = nn.Sequential(OrderedDict(block))

        self.output_block = nn.Sequential(OrderedDict([
            ('conv', ConvLayer(n_featuremap(self.resolution_log2-8), n_featuremap(1), 3, use_wscale=use_wscale)),
            ('view', View(-1)),
            ('act0', self.act),
            ('dense0', FCLayer(fmap_base, n_featuremap(0), use_wscale=use_wscale)),
            ('act1', self.act),
            ('dense1', FCLayer(n_featuremap(0), 1, use_wscale=use_wscale))
        ]))

    def forward(self, x):
        if self.structure == 'fixed':
            # [1024, 1024, 3] --> [1024, 1024, n_featuremap(9)=16]
            x = self.act(self.fromrgb(x))   
            # [1024, 1024, 16] --> [4, 4, 512]
            x = self.blocks(x)
            # [4, 4, 512] -- > [1]
            output = self.output_block(x)

        return output

if __name__ == "__main__":
    GNet = styleGenerator().to('cuda')
    DNet = styleDiscriminator().to('cuda')
    z_data = torch.randn((2, 512), dtype=torch.float32, device='cuda')
    fake_img = GNet(z_data)
    dis_op = DNet(fake_img)