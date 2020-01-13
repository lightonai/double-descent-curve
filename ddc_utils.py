import os

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self, kernel_depth, channel_input=3, kernel_step=5, stride=1, padding=0, beta=1.):
        super(Net, self).__init__()
        self.encoder = nn.Conv2d(in_channels=channel_input, out_channels=kernel_depth, kernel_size=kernel_step,
                                 stride=stride, padding=padding)
        self.decoder = nn.ConvTranspose2d(in_channels=kernel_depth, out_channels=channel_input, kernel_size=kernel_step,
                                          stride=stride, padding=padding)
        self.beta = beta
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.encoder.weight, gain=5 / 3)
        nn.init.xavier_uniform_(self.decoder.weight, gain=5 / 3)

    def encoder(self, images):
        h = (torch.sign(self.encoder(images)) + 1) / 2
        return h


def ae_post_processing(x):
    x = x.squeeze(1)
    return x.int()


def mnist(batch_size, n_train_samples=None, binary=False, encoder='threshold'):
    """
    return MNIST train and test dataloader with given batch_size.
    Shuffle is set true for train and false for test.
    Normalization is applied to both
    """
    if encoder not in {'threshold', 'autoencoder'}:
        raise ValueError("Known binary methods are 'threshold', 'autoencoder'")
    if binary:
        if encoder == 'threshold':
            threshold = transforms.Lambda(lambda x: (x > 28 / 255))
            t = transforms.Compose([transforms.ToTensor(), threshold])
        if encoder == 'autoencoder':
            pwd = os.path.dirname(os.path.abspath(__file__))
            net = Net(kernel_depth=24, channel_input=1, kernel_step=6, stride=2, padding=2)
            path = pwd + '/ae_weights/net_mnist.pt'
            net.load_state_dict(torch.load(path))
            net.eval()
            ae = transforms.Lambda(lambda x: net.encoder(x))
            un = transforms.Lambda(lambda x: x.unsqueeze(0))
            t = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), un, ae, ae_post_processing])
    else:
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=t)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=t)
    if n_train_samples:
        indices = list(range(len(trainset)))
        np.random.seed(1234)
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        mnist_dl_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
        mnist_dl_test = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    else:
        mnist_dl_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        mnist_dl_test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return mnist_dl_train, mnist_dl_test


def cifar10(batch_size, n_train_samples=None, binary=False, encoder='threshold'):
    """
    Return cifar10 train and test dataloader with given batch_size.
    Shuffle is set true for train and false for test.
    Normalization is applied to both
    """
    if encoder not in {'threshold', 'autoencoder'}:
        raise ValueError("Known binary methods are 'threshold', 'autoencoder'")
    if binary:
        if encoder == 'threshold':
            threshold = transforms.Lambda(lambda x: (x > 32 / 255))
            t = transforms.Compose([transforms.ToTensor(), threshold])
        if encoder == 'autoencoder':
            pwd = os.path.dirname(os.path.abspath(__file__))
            net = Net(kernel_depth=24, channel_input=3, kernel_step=6, stride=2, padding=2)
            path = pwd + '/ae_weights/net_cifar10.pt'
            net.load_state_dict(torch.load(path))
            net.eval()
            ae = transforms.Lambda(lambda x: net.encoder(x))
            un = transforms.Lambda(lambda x: x.unsqueeze(0))
            t = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), un, ae,
                 ae_post_processing])

    else:
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=t)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=t)
    if n_train_samples:
        indices = list(range(len(trainset)))
        np.random.seed(1234)
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        c10_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
        c10_test = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    else:
        c10_train = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
        c10_test = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)
    return c10_train, c10_test
