import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import os
import tqdm
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def parse_args():
    parser = argparse.ArgumentParser('StyleGAN')
    parser.add_argument('--ver', type=int, default=2, choices=[1, 2],
                        help='the version of the network, [1, 2]. default=1')
    parser.add_argument('--ds', type=str, default='ffhq',
                        help='which dataset you want to train. default="ffhq"')
    parser.add_argument('--bs', type=int, default=8,
                        help='batch size. default=8')
    parser.add_argument('--nw', type=int, default=4, 
                        help='number of worker using in dataloader: default=4')
    parser.add_argument('--lrD', type=float, default=0.00005, 
                        help='adam: learning rete for discriminator: default=0.00005. 1e-4 for wgan-gp in default')
    parser.add_argument('--lrG', type=float, default=0.00005, 
                        help='adam: learning rete for generator: default=0.00005. 1e-4 for wgan-gp in default')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='size of image')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='channels of image')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='number of D iters per each G iters: default=5')
    parser.add_argument('--loss_type', type=str, default='wgan_gp', choices=['vanilla', 'wgan', 'wgan_gp', 'vanilla_R12P'],
                        help='type of the loss: vanilla, wgan, wang_gp, vanilla_R12P. default=wgan_gp')
    parser.add_argument('--gp_factor', type=int, default=10.,
                        help='train wgan-gp with the gradient penalty coefficient: default=0')
    parser.add_argument('--clamp', type=float, default=0.01,
                        help='number to clamp the parameters: default=0.01')
    parser.add_argument('--s_epoch', type=int, default=1,
                        help='the starting epoch: default=1')
    parser.add_argument('--e_epoch', type=int, default=25,
                        help='the ending epoch: default=25')
    parser.add_argument('--z_dim', type=int, default=512,
                        help='dimensionality of the input features for generator: default=512')
    parser.add_argument('--w_dim', type=int, default=512,
                        help='dimensionality of the intermediate features W for generator: default=512')
    parser.add_argument('--save_img_iter', type=int, default=500,
                        help='save generation image every iteration: default=500')
    parser.add_argument('--save_ck_epoch', type=int, default=1,
                        help='save training checkpoints every epoch. default=1')

    parser.add_argument('--log_dir', type=str, default='log',
                        help='path to save the logger: default=log')
    parser.add_argument('--ck_dir', type=str, default='checkpoints',
                        help='path to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='path to save the results')
    parser.add_argument('--exp_ver', type=str, default='test',
                        help='experiment version')

    return parser.parse_args()

def create_folder(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)

def cal_num_param(m):
    num_param = 0
    for p in m.parameters():
        num_param += p.numel()
    return round(num_param / 1e6, 2)

def train():
    # create folders
    create_folder(os.path.join(args.result_dir, args.ds, args.exp_ver, args.loss_type, 'images'))
    create_folder(os.path.join(args.ck_dir, args.ds, args.exp_ver, args.loss_type))
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir,  args.ds, args.exp_ver))

    # prepare the data
    trainSet = datasets.ImageFolder(
        root='../datasets/'+args.ds,
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    )
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    print('Finished data preparation!!! Number of training images is {0}!!!'.format(len(trainLoader)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # construct the model
    netG = styleGenerator(z_dim=args.z_dim, w_dim=args.w_dim, img_resolution=args.img_size, img_channels=args.img_channels).to(device)
    netD = styleDiscriminator(img_resolution=args.img_size, img_channels=args.img_channels).to(device)
    print(f'Finished network construction!!! \nParameters of GNet is {cal_num_param(netG)} M, DNet is {cal_num_param(netD)} M!!!')

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lrD, betas=(0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lrG, betas=(0, 0.9)) 

    fixed_noise = torch.randn(64, args.z_dim, device=device)
    one = torch.FloatTensor([1]).to(device)   # to adjust the gradient  
    mone = one * -1

    print(f'Begin to train: on {device} and {args.loss_type} loss....')
    gen_trained = False; gen_iterations = 0; img_list = []
    for epoch in range(args.s_epoch, args.e_epoch+1):
        for i, data in enumerate(tqdm.tqdm(trainLoader)):
            # train discriminator
            # import ipdb; ipdb.set_trace()
            optimizerD.zero_grad()

            data_real = data[0].to(device)
            errD_real = netD(data_real).mean()

            noise = torch.randn(data_real.size(0), args.z_dim, device=device)
            data_fake = netG(noise)
            errD_fake = netD(data_fake.detach()).mean()

            errD_real.backward(mone, retain_graph=False)
            errD_fake.backward(one, retain_graph=False)

            if args.loss_type == 'wgan_gp':
                alpha = torch.rand(data_real.size(0), 1, 1, 1, device=device)
                x_hat = alpha * data_real + (1 - alpha) * data_fake.detach()
                # x_hat = alpha * data_real.data + (1 - alpha) * data_fake.data
                x_hat.requires_grad = True
                pred_hat = netD(x_hat)
                gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones_like(pred_hat, device=device), 
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = args.gp_factor * ((gradients.norm(2, 1) - 1) ** 2).mean()
                gradient_penalty.backward()
            else:
                gradient_penalty = 0
            
            errD = errD_real - errD_fake + gradient_penalty
            optimizerD.step()

            # train generator
            if i % args.n_critic == 0:
                optimizerG.zero_grad()
                noise = torch.randn(args.bs, args.z_dim, device=device)
                data_fake = netG(noise)
                errG = -netD(data_fake).mean()     # pay attention to the negative sign
                errG.backward(one)
                optimizerG.step()
                # flag
                gen_iterations += 1
                gen_trained = True

            writer.add_scalars('loss', {'errD': errD.item(), 'errG': errG.item()}, gen_iterations)
            if gen_trained and gen_iterations == 1 or gen_iterations % 50 == 0:
                print('[Epoch]:{:d}/{:d} \t[ite_D]:{:d}/{:d} \t[iter_G]:{:d} \t[Loss_D]:{:.4f} \t[Loss_G]:{:.4f}'.format(
                        epoch, args.e_epoch, i+1, len(trainLoader), gen_iterations, errD.item(), errG.item()))
            if gen_trained and gen_iterations == 1 or gen_iterations % args.save_img_iter == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(img)
                vutils.save_image(img[None,...].clone(), os.path.join(args.result_dir, args.ds, args.exp_ver, args.loss_type, 'images', f'{gen_iterations}.png'))

            if gen_trained:
                gen_trained = False
                
        if epoch % args.save_ck_epoch == 0:
            save_ck = {
                'gen': netG.state_dict(),
                'dis': netD.state_dict()
            }
            torch.save(save_ck, os.path.join(args.ck_dir, args.ds, args.exp_ver, args.loss_type, f'checkpoints_{epoch}_epoch.pth'))
        
    writer.close()
    print('Finished training!!!')
    # save animation generation results
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(f'generation results every {args.save_img_iter} training iterations')
    ims = [[plt.imshow(np.transpose(img, (1,2,0)), animated=True)] for img in img_list]
    im_ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    im_ani.save(os.path.join(args.result_dir, args.ds, args.exp_ver, args.loss_type, 'gen_process.mp4'))

if __name__ == '__main__':
    args = parse_args()
    if args.ver == 1:
        from models.styleGAN import styleGenerator, styleDiscriminator
    else:
        from models.styleGAN2 import styleGenerator, styleDiscriminator
    train()