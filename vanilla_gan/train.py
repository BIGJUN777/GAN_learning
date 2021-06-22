import os
import argparse
from tqdm import tqdm
import numpy as np
from functools import reduce

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import save_image 
from torchvision import datasets, transforms

from model import Generator, Discriminator

def train():
    args.exp_ver = 'conditional/'+args.exp_ver if args.cond_num_class else args.exp_ver
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='./log/' + args.exp_ver)
    if not os.path.exists('./checkpoints/' + args.exp_ver):
        os.makedirs('./checkpoints/' + args.exp_ver)
    if not os.path.exists('./results/' + args.exp_ver):
        os.makedirs('./results/' + args.exp_ver)
    # construct the data
    data_root = '../datasets'
    if not os.path.exists(data_root):
        os.makedirs(data_root, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        dataset = datasets.MNIST(
                    root = data_root,
                    train = True,
                    download = True,
                    transform = transforms.Compose(
                        [transforms.Resize(args.img_size), \
                         transforms.ToTensor(), \
                         transforms.Normalize([0.5,], [0.5])]
                    ),
                ),
        batch_size = args.bs,
        shuffle = True,
        num_workers = args.nw,
    )
    num_batches = len(dataloader)

    # initialize the model
    generator = Generator(args.z_dim, args.img_size**2, args.cond_num_class).to(device).train()
    discriminator = Discriminator(args.img_size**2, 1, args.cond_num_class).to(device).train()

    # set up optimizer && criterion
    criterion = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        i_n = 0
        for imgs, target in tqdm(dataloader):
            # import ipdb ;ipdb.set_trace()
            # train distriminator
            optimizer_d.zero_grad()
            # construce training data
            real_gt = torch.ones((imgs.shape[0], 1)).to(device)
            fake_gt = torch.zeros((imgs.shape[0], 1)).to(device)
            real_imgs = imgs.view(imgs.shape[0], -1).to(device)
            z_data = torch.randn(imgs.shape[0], args.z_dim).to(device)
            # training
            gen_imgs = generator(z_data, target).detach()
            real_loss = criterion(discriminator(real_imgs, target), real_gt)
            fake_loss = criterion(discriminator(gen_imgs, target), fake_gt)
            d_loss = (real_loss+fake_loss) / 2     
            d_loss.backward()
            optimizer_d.step()

            # train generator
            # if i%args.k_step == args.k_step-1: 
            optimizer_g.zero_grad()
            z_data = torch.randn(imgs.shape[0], args.z_dim).to(device)
            gen_imgs = generator(z_data, target)
            # NOTE: using real_gt to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs, target), real_gt)     
            g_loss.backward()
            optimizer_g.step()
            # logger
            iter_num = epoch*num_batches+i_n
            writer.add_scalars('train_loss', \
                               {'d_loss': d_loss.data, 'g_loss': g_loss.data}, \
                               iter_num,
            )
            if iter_num % args.sample_interval == args.sample_interval-1:
                save_image(gen_imgs[:25].view(25, 1, args.img_size, args.img_size), \
                           f'./results/{args.exp_ver}/{iter_num}_{reduce(lambda x,y:x+y, map(str, np.array(target[:25])))}.png', nrow=5, normalize=True
                )
                # save_image(imgs[:25].view(25, 1, args.img_size, args.img_size), \
                #            f'./results/{args.exp_ver}/{iter_num}_gt.png', nrow=5, normalize=True
                # )

            i_n += 1
        if epoch>=150 and epoch%10==9:
            save_info = {'args': args, 
                         'checkpoints_g': generator.state_dict(), 
                         'checkpoints_d': discriminator.state_dict(),
            }
            save_name = os.path.join('./checkpoints', args.exp_ver, 'checkpoint_'+str(epoch+1)+'_epoch.pth')
            torch.save(save_info, save_name)
    writer.close()
    print('Finish training!!!')
        

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='number of epoch for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rete')
    parser.add_argument('--bs', type=int, default=64, help='sizes of the batch')
    parser.add_argument('--nw', type=int, default=0, help='number of worker using in dataloader')
    parser.add_argument('--z_dim', type=int, default=100, help='dimensionality of input features for generator')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image')
    parser.add_argument('--k_step', type=int, default=1, help='number of step to apply to train the discriminator')
    parser.add_argument('--exp_ver', type=str, default='test', help='version name of the experiment')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval between saving generated images')
    parser.add_argument('--cond_num_class', type=int, default=0, help='if using conditonal GAN, set the number of the class, default: 0')

    return parser.parse_args()

if __name__ == '__main__':
    args = argument()
    train()