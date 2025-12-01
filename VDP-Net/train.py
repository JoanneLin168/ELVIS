import sys, os
import numpy as np
import torch
from datetime import datetime
import argparse, json
import random

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataset.degradation_dataset import DegradationVideoDataset
from src.models import build_model
from src.engine.VDPNet_trainer import VDPNetTrainer

os.environ["USE_LIBUX"] = "0"

def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_arguments():
    parser = argparse.ArgumentParser(description='DEN noise model training options.')
    parser.add_argument('--dataset_root', default="./data", help='Path to dataset')
    parser.add_argument('--output_folder', default = './runs/', help='Specify where to save checkpoints during training')

    parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint to load')
    parser.add_argument('--num_frames', default=5, type=int, help='Number of frames to sample from each video')
    parser.add_argument('--w1', default=1, type=float, help='Weight for MLP loss')
    parser.add_argument('--w2', default=1, type=float, help='Weight for cosine angle loss')
    
    parser.add_argument('--lr', default = 0.0002, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--batch_size', default = 2, type=int)
    parser.add_argument('--patch_size', default=256, type=int) 
    parser.add_argument('--eval_patch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--display_freq', default=500, type=int, help='Frequency of visualizing training results')
    parser.add_argument('--log_freq', default=100, type=int, help='Frequency of logging training results')
    parser.add_argument('--save_freq', default=5, type=int, help='Frequency of saving checkpoints')

    parser.add_argument('--device', default= 'cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
    parser.add_argument('--notes', default='notes', type=str, help='Add notes to the experiment')

    args = parser.parse_args()

    return args
    
def main(args):

    # Misc setup
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    # output_folder = args.output_folder + 'den_' + start_time + '/'
    output_folder = args.output_folder + 'den_' + args.notes + '/' + start_time + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder + 'checkpoints')

    with open(output_folder + 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    args.output_folder = output_folder
    writer = SummaryWriter(args.output_folder)
        
    # Training setup
    train_dir = os.path.join(args.dataset_root, 'train_all_frames', 'JPEGImages')
    val_dir = os.path.join(args.dataset_root, 'valid_all_frames', 'JPEGImages')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomCrop((args.patch_size, args.patch_size), pad_if_needed=True), # crop will be done in the dataset class
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((args.patch_size, args.patch_size)),
    ])

    train_set = DegradationVideoDataset(train_dir, args.num_frames, args.patch_size, train=True, transform=train_transform)
    val_set = DegradationVideoDataset(val_dir, args.num_frames, args.patch_size, train=False, transform=val_transform)

    # Model setup
    model = build_model(args)
    model.to(args.device)

    # Create trainer
    trainer = VDPNetTrainer(args, model, train_set, val_set, writer)
    trainer.run()

    writer.close()
    print("Finished training!")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')

    args = get_arguments()
    set_seed(args.seed)
    main(args)
