"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model
import matplotlib.pyplot as plt

def train(vae,trainloader,optimizer,device):
    vae.train()  # set to training mode
 
    total_loss = []
    for data in trainloader:
        inputs,_ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        loss = vae(inputs)
        loss.backward()
        optimizer.step()
        total_loss.append(-loss.item()/len(inputs))
    mean_loss = np.mean(total_loss)
    return mean_loss


def test(vae, testloader,device):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        total_loss = []
        for data in testloader:
            inputs, _ = data
            inputs =  inputs.to(device)
            loss = vae(inputs)
            total_loss.append(-loss.item()/len(inputs))
        mean_elbo = np.mean(total_loss)
        return mean_elbo


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    total_loss_train = []
    total_loss_val= []
    for epoch in range(args.epochs):
        train_loss = train(vae,trainloader,optimizer,device)
        print('epoch %d/%5d -- train_elbo: %.3f' % (epoch + 1, args.epochs, train_loss))     
        total_loss_train.append(train_loss)
        
        val_loss = test(vae,testloader,device)
        print('epoch %d/%5d -- test_elbo: %.3f' % (epoch + 1, args.epochs, val_loss))
        total_loss_val.append(val_loss)
        
    plt.figure()        
    plt.plot(total_loss_train, linewidth=3, color='blue', label='train loss')
    plt.plot(total_loss_val, linewidth=3, color='orange', label='test loss')
    plt.legend()
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.grid(True)
    plt.savefig('ELBO.png')

    
    sample_size=64
    with torch.no_grad():
        samples = vae.sample(sample_size).cpu()
        samples.clamp_(0,1)
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + '.png')


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
