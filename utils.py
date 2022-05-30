import torch 
from torchvision import datasets, transforms 
from torch.utils.data import Dataset 
import os  


def image_preprocess_transform():
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.RandomRotation(5),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(pretrained_size, padding=10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                        ])

    test_transform = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                        ]) 
    return train_transform, test_transform 



def dataset_load(args): 
    if 'cifar' in args.dataset_type: 
        data_path = os.path.join(args.data_path, 'cifar') 
    else:
        data_path = os.path.join(args.data_path, 'imagenet')  

    if args.dataset_type == 'cifar-10': 
        train_loader, test_loader = cifar10_data_load(data_path, args.batch_size) 
    elif args.dataset_type == 'cifar-100': 
        train_loader, test_loader = cifar100_data_load(data_path, args.batch_size) 
    else:
        print('imagenet dataset not implemented!')
    return train_loader, test_loader 



def cifar10_data_load(data_path, batch_size, distribution=False):
    # image transform 
    train_transform, test_transform = image_preprocess_transform() 

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    train_set = datasets.CIFAR10(data_path, train=True, download=False, transform=train_transform)
    
    test_set = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)

    if distribution == False:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False) # num_workers=2 
    else: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set)
        val_sampler = torch.utils.data.sampler.SequentialSampler(test_set) 
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8,
                                                  sampler=train_sampler) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=8,
                                                  sampler=val_sampler) 

    return train_loader, test_loader



def cifar100_data_load(data_path, batch_size, distribution=False):
    # image transform 
    train_transform, test_transform = image_preprocess_transform() 


    train_set = datasets.CIFAR100(data_path, train=True, download=False, transform=train_transform)
    
    test_set = datasets.CIFAR100(data_path, train=False, download=False, transform=test_transform)

    if distribution == False:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False) # num_workers=2 
    else: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set)
        val_sampler = torch.utils.data.sampler.SequentialSampler(test_set) 
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8,
                                                  sampler=train_sampler) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=8,
                                                  sampler=val_sampler) 

    return train_loader, test_loader