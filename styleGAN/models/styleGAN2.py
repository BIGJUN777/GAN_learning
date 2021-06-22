import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np

def add_path(path=None):
    if path == None:
        import os
        import sys
        # path = os.path.realpath(__file__)
        # path = os.path.join(*path.split('/')[:-2])
        path = os.path.dirname(sys.path[0])
        sys.path.append(path)
    else:
        import sys
        sys.path.append(path)
add_path()

from models.modules import * 

    
class MappingNet(nn.Module):
    def __init__(self, z_dim=512, hid_dim=512, lrmul=0.01, activation='lrelu', use_wscale=True, bias=0):
        super(MappingNet, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('pixel_norm', PixelNormLayer()),
            ('dense0', FCLayer(z_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense1', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense2', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense3', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense4', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense5', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense6', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),
            ('dense7', FCLayer(hid_dim, hid_dim, activation=activation, lrmul=lrmul, use_wscale=use_wscale, bias=bias)),           
        ]))

    def forward(self, x):
        return self.net(x)


class GSynthesisNet(nn.Module):
    def __init__(self, 
                 w_dim,                      # Intermediate latent (W) dimensionality.
                 img_resolution,             # Output image resolution.
                 img_channels,               # Number of color channels.
                 channel_base = 32768,       # Overall multiplier for the number of channels.
                 channel_max  = 512,         # Maximum number of channels in any layer.
                 **block_kwargs,             # Arguments for SynthesisBlock.
    ):

        super(GSynthesisNet, self).__init__()
        n_featuremap = lambda stage: min(int(channel_base / (2.0 ** stage)), channel_max)

        self.resolution_log2 = int(np.log2(img_resolution))
        blocks = []; self.num_conv = 0
        for res in range(2, self.resolution_log2+1):
            in_channel = n_featuremap(res-1) if res>2 else 0
            out_channel = n_featuremap(res)
            name = '{s}x{s}'.format(s=2**res)
            is_last = (res == self.resolution_log2)
            blocks.append((name, GSynthesisBlock(in_channel, out_channel, img_channels, 2**res, w_dim, is_last=is_last, **block_kwargs)))
            self.num_conv += blocks[-1][-1].num_conv
            if is_last:
                self.num_conv += blocks[-1][-1].num_torgb
        self.net = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, ws):
        assert ws.size(1) == self.num_conv
        block_ws = {}
        w_idx = 0
        for res in range(2, self.resolution_log2+1):
            name = '{s}x{s}'.format(s=2**res)
            # If the arch == 'skip', the W to the 'tRGB' is the same as the one to the first layer of the next block.
            # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d4b2afe9c27e3c305b721bc886d2cb5229458eba/training/networks.py#L465
            block_ws[name] = ws.narrow(1, w_idx, self.net[name].num_conv+self.net[name].num_torgb)
            w_idx += self.net[name].num_conv 
        
        x = img = None 
        for name, net in self.net.items():
            cur_ws = block_ws[name]
            x, img = net(x, cur_ws, img)

        return img


class styleGenerator(nn.Module):
    def __init__(self, 
                 z_dim,                      # Input latent (Z) dimensionality.
                 w_dim,                      # Intermediate latent (W) dimensionality.
                 img_resolution,             # Output resolution.
                 img_channels,               # Number of output color channels.
                 truncation_psi = 0.7,       # Trunction: threshold
                 truncation_cutoff = 8,      # Trunction: max layers
                 dlatent_avg_beta = 0.995,   # Trunction: coefficient to calculate the average of W
                 mapping_kwargs = {},        # Arguments for MappingNetwork.
                 synthesis_kwargs = {},      # Arguments for SynthesisNetwork.
    ):
        super(styleGenerator, self).__init__()
        self.mapping = MappingNet(z_dim, w_dim, **mapping_kwargs)
        self.synthesis = GSynthesisNet(w_dim, img_resolution, img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_conv
        if truncation_psi != 1:
            self.trunction = Truncation(torch.zeros(w_dim), truncation_cutoff, truncation_psi, dlatent_avg_beta)
        else:
            self.trunction = None

        assert truncation_cutoff < self.num_ws, 'the max layers to truncation must smaller than the number of latent W'
        self.tanh = nn.Tanh()

    def forward(self, z_latents):
        ws = self.mapping(z_latents)
        # Broadcast && truncate the latent W
        ws = ws.unsqueeze(1).expand(-1, self.num_ws, -1)
        if self.trunction is not None and self.training:
            self.trunction.update_avg(ws[:,0,:].detach())
            ws = self.trunction(ws)
        img = self.synthesis(ws)
        # TODO: if we need tanh function to scale the img to [-1, 1]
        # img = self.tanh(img)

        return img


class styleDiscriminator(nn.Module):
    def __init__(self, 
                img_resolution,                 # Input resolution.
                img_channels,                   # Number of input color channels.
                arch                = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
                channel_base        = 32768,    # Overall multiplier for the number of channels.
                channel_max         = 512,      # Maximum number of channels in any layer.
                block_kwargs        = {},       # Arguments for DiscriminatorBlock.
                epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.

    ):
        super(styleDiscriminator, self).__init__()
        n_featuremap = lambda stage: min(int(channel_base / (2.0 ** stage)), channel_max)

        self.resolution_log2 = int(np.log2(img_resolution))
        blocks = []
        for res in range(self.resolution_log2, 2, -1):
            in_channels = n_featuremap(res) if res < self.resolution_log2 else 0
            tmp_channels = n_featuremap(res)
            out_channels = n_featuremap(res-1)
            name = '{s}x{s}'.format(s=2**res)
            blocks.append((name, DiscriminatorBlock(in_channels, tmp_channels, out_channels, img_channels, arch, **block_kwargs)))

        self.blocks = nn.ModuleDict(OrderedDict(blocks))
        self.output_block = DiscriminatorEpilogue(n_featuremap(2), 4, img_channels, arch=arch, **epilogue_kwargs)

    def forward(self, img):
        x = None
        for name, net in self.blocks.items():
            # import ipdb; ipdb.set_trace()
            x, img = net(x, img)
        out = self.output_block(x, img)
        return out


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    GNet = styleGenerator(z_dim=512, w_dim=512, img_resolution=512, img_channels=3).to('cuda')
    DNet = styleDiscriminator(img_resolution=512, img_channels=3).to('cuda')
    data = torch.randn((2, 512), dtype=torch.float32, device='cuda')
    fake_img = GNet(data)
    out = DNet(fake_img)
    




    




