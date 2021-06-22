import torch
import torchvision
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import *

def parse_arg():
    parser = argparse.ArgumentParser('WGAN')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='dimensionality of the input features for generator: default=100')
    parser.add_argument('--s_epoch', type=int, default=1,
                        help='starting epoch: default=1')
    parser.add_argument('--e_epoch', type=int, default=25,
                        help='ending epoch: default=25')
    parser.add_argument('--lrD', type=float, default=0.00005, 
                        help='adam: learning rete for discriminator: default=0.00005. 1e-4 for wgan-gp in default')
    parser.add_argument('--lrG', type=float, default=0.00005, 
                        help='adam: learning rete for generator: default=0.00005. 1e-4 for wgan-gp in default')
    parser.add_argument('--bs', type=int, default=64, 
                        help='sizes of the batch: default=64')
    parser.add_argument('--nw', type=int, default=4, 
                        help='number of worker using in dataloader: default=4')
    parser.add_argument('--img_size', type=int, default=64,
                        help='size of image')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='number of D iters per each G iters: default=5')
    parser.add_argument('--clamp', type=float, default=0.01,
                        help='number to clamp the parameters: default=0.01')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='path to save the logger: default=log')
    parser.add_argument('--ck_dir', type=str, default='checkpoints',
                        help='path to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='path to save the results')
    parser.add_argument('--exp_ver', type=str, default='test',
                        help='experiment version')
    parser.add_argument('--save_img_iter', type=int, default=500,
                        help='save generation image every iteration: default=500')
    parser.add_argument('--save_ck_epoch', type=int, default=1,
                        help='save training checkpoints every epoch')
    parser.add_argument('--gp_factor', type=int, default=0,
                        help='train wgan-gp with the gradient penalty coefficient: default=0')

    return parser.parse_args()

def create_folder(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)

def dumps_json(obj, file_name):
    data = json.dumps(obj, sort_keys=True, indent=4)
    with open(file_name, 'w') as f:
        if isinstance(data, list):
            for i in data:
                f.write(str(i), ' \n')
        else:
            f.write(data)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)        

def train():
    # create the folders
    create_folder(os.path.join(args.result_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver, 'images'))
    create_folder(os.path.join(args.ck_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver))
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver))
    dumps_json(vars(args), os.path.join(args.ck_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver, 'train_info.json'))

    # prepare data
    dataset = torchvision.datasets.ImageFolder(
        root = '../datasets/celeba',
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.img_size),
            torchvision.transforms.CenterCrop(args.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    print('Finished data preparation!!!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # construct the model
    netG = Generator(args.z_dim).to(device)
    netD = Distriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    print('Finished network construction!!!')

    # setup optimizer and criterion
    if args.gp_factor:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lrD, betas=(0, 0.9))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lrG, betas=(0, 0.9))
    else:
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lrD)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lrG)

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)
    one = torch.FloatTensor([1]).to(device)   # to adjust the gradient  
    mone = one * -1

    print('Training...')
    gen_iterations = 0; img_list = []
    for epoch in range(args.s_epoch, args.e_epoch+1):
        data_iter = iter(dataloader)
        data_used_num = 0
        while data_used_num < len(dataloader):
            # if gen_iterations == 15: break
            gen_iterations += 1
            # To train the discriminator
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_critic = args.n_critic   # 100
            else:
                n_critic = args.n_critic
            Diter = 0
            while Diter < n_critic and data_used_num < len(dataloader):
                Diter += 1
                data_used_num += 1
                # netD.zero_grad()
                optimizerD.zero_grad()
                # clamp the parameters to a cube when training wgan
                if not args.gp_factor:
                    for p in netD.parameters():
                        p.data.clamp_(-args.clamp, args.clamp)
                    gradient_penalty = 0
                
                real_data, _ = data_iter.next()
                real_data = real_data.to(device)
                errD_real = netD(real_data).mean()
                
                noise = torch.randn(real_data.size(0), args.z_dim, 1, 1, device=device)
                fake_data = netG(noise)
                errD_fake = netD(fake_data).mean()

                # The code below is following the persudo code in the paper.
                errD_real.backward(mone, retain_graph=False)
                errD_fake.backward(one, retain_graph=False)

                # training wgan-gp
                if args.gp_factor:
                    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
                    # pay attention to detach(). If without detach(), you need to set `retain_graph=True` in the backward() in L148-L149
                    # x_hat = alpha * real_data + (1 - alpha) * fake_data
                    x_hat = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
                    x_hat.requires_grad = True
                    pred_hat = netD(x_hat)
                    gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones_like(pred_hat, device=device), 
                                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                    # import ipdb; ipdb.set_trace()
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = args.gp_factor * ((gradients.norm(2, 1) - 1) ** 2).mean()
                    gradient_penalty.backward()

                errD = errD_real - errD_fake + gradient_penalty

                # because the optimizer always tries to minimize the loss, 
                # so the code below is the straightforward way to update the weights
                # !NOTE: the following errD is the negative wasserstein distance
                # errD = errD_fake - errD_real
                # errD.backward() 
                optimizerD.step()
            
            # To train the generator
            optimizerG.zero_grad()
            noise = torch.randn(args.bs, args.z_dim, 1, 1, device=device)
            fake_data = netG(noise)
            errG = -netD(fake_data).mean()     # pay attention to the negative sign
            errG.backward(one)
            optimizerG.step()
            
            writer.add_scalars('loss', {'-errD': errD.item(), 'errG': errG.item()}, gen_iterations)
            if gen_iterations == 1 or gen_iterations % 50 == 0:
                print('[Epoch]:{:d}/{:d} \t[ite_D]:{:d}/{:d} \t[iter_G]:{:d} \t[Loss_D]:{:.4f} \t[Loss_G]:{:.4f}'.format(
                        epoch, args.e_epoch, data_used_num, len(dataloader), gen_iterations, errD.item(), errG.item()))
            if gen_iterations == 1 or gen_iterations % args.save_img_iter == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(img)
                vutils.save_image(img[None,...].clone(), os.path.join(args.result_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver, 'images', f'{gen_iterations}.png'))
                
        if epoch % args.save_ck_epoch == 0:
            save_ck = {
                'gen': netG.state_dict(),
                'dis': netD.state_dict()
            }
            torch.save(save_ck, os.path.join(args.ck_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver, f'checkpoints_{epoch}_epoch.pth'))
        
    writer.close()
    print('Finished training!!!')
    # save animation generation results
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(f'generation results every {args.save_img_iter} training iterations')
    ims = [[plt.imshow(np.transpose(img, (1,2,0)), animated=True)] for img in img_list]
    im_ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    im_ani.save(os.path.join(args.result_dir, 'wgan-gp' if args.gp_factor else 'wgan', args.exp_ver, 'gen_process.mp4'))

if __name__ == '__main__':
    args = parse_arg()
    train()