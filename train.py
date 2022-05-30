from PIL import Image 
import torch 
import torch.nn as nn 
from model import ViTConfig, ViTForImageClassification, DeltaViTForImageClassification
from utils import dataset_load 
import argparse 
import os 
from tqdm import tqdm 
import random 
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

SEED = 2022

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(train_loader, model, optimizer, loss_fn, epoch, args): 
    model.train() 
    running_loss = 0.0 
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar:
        for it, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device) 
            optimizer.zero_grad()
            out = model(image)[0]
            loss = loss_fn(out, label) 
            loss.backward() 
            optimizer.step()

            running_loss += loss.item() 
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update() 
            if it % 10000 == 0:
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    # 'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.output_path, "latest.pth"),)
            break 



def validation(test_loader, model, epoch): 
    model.eval() 
    acc = .0 
    time_stamp = 0
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_loader)) as pbar:
        for it, (image, label) in enumerate(test_loader): 
            image, label = image.to(device), label.to(device) 
            with torch.no_grad(): 
                out = model(image)[0]  # (bsz, vob)
                predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                acc += (predict_y == label).sum().item() / predict_y.size(0)
            pbar.set_postfix(acc=acc / (it + 1))
            pbar.update() 
            time_stamp += 1 
            break 
    val_acc = acc / time_stamp 
    return val_acc 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data') 
    parser.add_argument('--dataset_type', default='cifar-10') 
    parser.add_argument('--vit_path', default='./ckpt/vit') 
    parser.add_argument('--output_path', default='./ckpt/delta-vit') 
    parser.add_argument('--batch_size', default=5) 
    parser.add_argument('--epochs', default=100) 
    parser.add_argument('--perturb_flag', default=True) 
    parser.add_argument('--load_from_last', default=True) 
    args = parser.parse_args()

    # load dataset 
    train_loader, test_loader = dataset_load(args) 

    # load model and optimizer 
    config = ViTConfig(num_labels=100) 
    if args.perturb_flag == False:
        model = ViTForImageClassification(config) 
    else:
        model = DeltaViTForImageClassification(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  


    if args.load_from_last == True: 
        fname = os.path.join(args.output_path, 'latest.pth') 
        if os.path.exists(fname): 
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            # torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optimizer.load_state_dict(data['optimizer']) 
            print('load last trained ckpt!')
    else:
        model.vit.from_pretrained(args.vit_path) 

    model = model.to(device) 
    loss_fn = nn.CrossEntropyLoss()  


    for epoch in range(args.epochs): 
        train(train_loader, model, optimizer, loss_fn, epoch, args) 
        validation(test_loader, model, epoch)
        break 


if __name__ == '__main__': 
    main()