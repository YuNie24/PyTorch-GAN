import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch
# from utils.fashion_mnist import MNIST, FashionMNIST

def get_data_loader(args):

    # if args.dataset == 'mnist':
    #     trans = transforms.Compose([
    #         transforms.Scale(32),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #     train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
    #     test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)
    #
    # elif args.dataset == 'fashion-mnist':
    #     trans = transforms.Compose([
    #         transforms.Scale(32),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #     train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
    #     test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)
    image_size = 64
    batch_size = 128
    workers = 2

    if args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'bedroom':
        dataroot = "/home/user/workspace/ykim/Generative-Adversarial-Networks-Cookbook/data0/lsun"

        train_dataset = dset.ImageFolder(root=dataroot,
                                       transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        print("bedroom dataset: ", train_dataset)

    # Check if everything is ok with loading datasets
    assert train_dataset
    # assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    test_dataloader  = data_utils.DataLoader(train_dataset,  batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    return train_dataloader, test_dataloader
