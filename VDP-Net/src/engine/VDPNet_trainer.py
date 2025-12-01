import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tqdm
import time
import torchvision

from src.engine.trainer import BaseTrainer
from src.engine import utils

def cosine_angle_loss(pred_deg, target_deg, mean=True):
    # Convert to radians from [0, 1]
    pred_rad = pred_deg * torch.pi
    target_rad = target_deg * torch.pi

    if mean:
        return (1-torch.cos(torch.abs(pred_rad - target_rad))).mean()
    else:
        return (1-torch.cos(torch.abs(pred_rad - target_rad)))

class VDPNetTrainer(BaseTrainer):
    def __init__(self, args, model, train_set, val_set, writer):
        super().__init__(args, model, train_set, val_set, writer)
        self.eval_patch_size = args.eval_patch_size
        self.num_frames = args.num_frames
        self.best_mlp = 1e6
        self.w1 = args.w1
        self.w2 = args.w2

        self.losses = {'mlp_loss': [], 'angle_loss': [], 'total_loss': []}
        self.noise_labels = ['exposure_value', 'shot_noise_log',
                            'read_noise', 'quant_noise', 'band_noise', 'band_noise_angle',
                            'blur_sigma1', 'blur_sigma2', 'blur_angle']

        self.noise_losses = {label: [] for label in self.noise_labels}
        
        # Load checkpoint if specified
        if args.checkpoint:
            self.load_from_checkpoint(args.checkpoint)
        
        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers)
        
        self.val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.curr_epoch = checkpoint['epoch'] + 1
        self.total_iter = checkpoint['total_iter'] + 1
        self.best_mlp = checkpoint['avg_mlp']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set best_kld from checkpoint if available
        if 'best' in checkpoint_path:
            self.best_mlp = checkpoint.get('avg_mlp', 1e6)
        else:
            try:
                best_checkpoint = torch.load(self.folder_name + 'checkpoints/best.pt')
                self.best_mlp = best_checkpoint.get('avg_mlp', 1e6)
            except FileNotFoundError:
                print("No best checkpoint found. Starting with best KLD from checkpoint.")
        
        print(f"Loaded checkpoint from {checkpoint_path}. Will train from epoch {checkpoint['epoch'] + 1}.")

    
    def save_debug_images(self, noisy_frames, clean_frames, gt_labels):
        # Save the first batch of noisy frames, clean frames, and ground truth labels for debugging
        B, N, C, H, W = noisy_frames.shape

        # Save noisy frames
        gt_labels_path = os.path.join(self.folder_name, 'debug', f'epoch_{self.curr_epoch}', 'gt_labels.npy')
        
        for i in range(self.num_frames):
            noisy_path = os.path.join(self.folder_name, 'debug', f'epoch_{self.curr_epoch}', 'input', f'{i:04d}.png')
            clean_path = os.path.join(self.folder_name, 'debug', f'epoch_{self.curr_epoch}', 'GT', f'{i:04d}.png')
            os.makedirs(os.path.dirname(noisy_path), exist_ok=True)
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)
            torchvision.utils.save_image(noisy_frames[0, i], noisy_path)
            torchvision.utils.save_image(clean_frames[0, i], clean_path)
        np.save(gt_labels_path, gt_labels[0].cpu().numpy())

    def train(self):
        total_loss = 0
        self.model.train()
        for i, sample in enumerate(tqdm.tqdm(self.train_loader, desc=f'Training', leave=False)):

            noisy_frames = sample['noisy_frames'].to(self.device)
            clean_frames = sample['clean_frames'].to(self.device)
            clean_center = sample['clean_center'].to(self.device)
            gt_labels = sample['gt_labels'].to(self.device)

            self.optimizer.zero_grad()

            B, N, C, H, W = noisy_frames.shape

            # Generate a batch of images
            output = self.model(clean_frames, noisy_frames)
            synth_noisy_full = output['synth_noisy']
            pred_labels = output['pred_labels']

            # Patchify the images if needed
            noisy_frames = noisy_frames.view(B * N, C, H, W)
            clean_frames = clean_frames.view(B * N, C, H, W)
            synth_noisy_full = synth_noisy_full.view(B * N, C, H, W)
            if self.patch_size > self.eval_patch_size:
                synth_patches = utils.split_into_patches2d(synth_noisy_full, self.eval_patch_size).to(self.device)
                noisy_patches = utils.split_into_patches2d(noisy_frames, self.eval_patch_size).to(self.device)
                clean_patches = utils.split_into_patches2d(clean_frames, self.eval_patch_size).to(self.device)
            else:
                synth_patches = synth_noisy_full
                noisy_patches = noisy_frames
                clean_patches = clean_frames
            noisy_frames = noisy_frames.view(B, N, C, H, W)
            clean_frames = clean_frames.view(B, N, C, H, W)
            synth_noisy_full = synth_noisy_full.view(B, N, C, H, W)

            # Display images during training (before FFT for visualization)
            if self.total_iter % self.display_freq == 0:
                gt_plt = noisy_frames.cpu().detach()[0, N//2]
                out_plt = synth_noisy_full.cpu().detach()[0, N//2]
                concatenated_images = torch.cat((gt_plt, out_plt), dim=2)
                concatenated_images = torch.clamp(concatenated_images, 0, 1)
                self.writer.add_image(f'Train/Synth_Images', concatenated_images, self.total_iter)
  

            # Loss
            mlp_loss = F.l1_loss(pred_labels[:, :, :-1], gt_labels[:, :, :-1]) # l1 loss for all parameters except blur angle
            angle_loss = cosine_angle_loss(pred_labels[:, :, -1], gt_labels[:, :, -1], mean=True) # motion blur angle loss
            loss = self.w1 * mlp_loss + self.w2 * angle_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Save losses for logging later
            self.losses['mlp_loss'].append(mlp_loss.item())
            self.losses['angle_loss'].append(angle_loss.item())
            self.losses['total_loss'].append(loss.item())

            # Save noise losses separately
            for i, label in enumerate(self.noise_labels):
                noise_loss = F.l1_loss(pred_labels[:, :, i], gt_labels[:, :, i])
                self.noise_losses[label].append(noise_loss.item())

            if self.total_iter % self.log_freq == 0:
                self.writer.add_scalar('Train/Total_Loss', np.round(np.mean(self.losses['total_loss']), 5), self.total_iter)
                self.writer.add_scalar('Train/MLP_Loss', np.round(np.mean(self.losses['mlp_loss']), 5), self.total_iter)
                self.writer.add_scalar('Train/Angle_Loss', np.round(np.mean(self.losses['angle_loss']), 5), self.total_iter)

                # Log the losses separately
                for label, noise_loss in zip(self.noise_labels, self.noise_losses.values()):
                    self.writer.add_scalar(f'Train/MLP_Loss/{label}', np.mean(noise_loss), self.total_iter)

                self.losses = {label: [] for label in self.losses} # Reset losses
                self.noise_losses = {label: [] for label in self.noise_labels}
            
            self.total_iter += 1

        print('Training - total loss:', total_loss)


    def validate(self):
        tot_kld = 0
        tot_mlp = 0
        self.model.eval()

        count = 0
        for sample in tqdm.tqdm(self.val_loader, desc='Validating', leave=False):
            with torch.no_grad():

                noisy_frames = sample['noisy_frames'].to(self.device)
                clean_frames = sample['clean_frames'].to(self.device)
                clean_center = sample['clean_center'].to(self.device)
                gt_labels = sample['gt_labels'].to(self.device)

                # Reshape input from [B, N, N, H, W] to [B*N, C, H, W]
                B, N, C, H, W = noisy_frames.shape
                
                output = self.model(clean_frames, noisy_frames)
                synth_noisy = output['synth_noisy']
                pred_labels = output['pred_labels'].view(B, N, -1)

                tot_mlp += F.l1_loss(pred_labels, gt_labels).item()

                # Patchify the images if needed
                noisy_frames = noisy_frames.view(B * N, C, H, W)
                clean_frames = clean_frames.view(B * N, C, H, W)
                synth_noisy = synth_noisy.view(B * N, C, H, W)
                if self.patch_size > self.eval_patch_size:
                    synth_noisy = utils.split_into_patches2d(synth_noisy, self.eval_patch_size).to(self.device)
                    real_noisy = utils.split_into_patches2d(noisy_frames, self.eval_patch_size).to(self.device)
                    clean_frames = utils.split_into_patches2d(clean_frames, self.eval_patch_size).to(self.device)
                else:
                    synth_noisy = synth_noisy
                    real_noisy = noisy_frames

                synth_noisemap = synth_noisy-clean_frames
                real_noisemap = real_noisy-clean_frames

                _, C, H, W = synth_noisemap.shape  # needed for when patchified
                synth_np = (synth_noisemap.view(-1, N, C, H, W)).detach().cpu().numpy()
                real_np = (real_noisemap.view(-1, N, C, H, W)).detach().cpu().numpy()
                kld_val = utils.cal_kld(synth_np, real_np)

                # Calculate validation metrics
                tot_kld += kld_val


        avg_kld = tot_kld/len(self.val_loader)
        avg_mlp = tot_mlp/len(self.val_loader)
        print(f'Validation - Average KLD: {avg_kld:.4f} MLP Loss: {avg_mlp:.4f}')
        self.writer.add_scalar('Validation/Average_KLD', np.round(avg_kld, 5), self.curr_epoch)
        self.writer.add_scalar('Validation/MLP_Loss', np.round(avg_mlp, 5), self.curr_epoch)

        # Save checkpoint
        checkpoint = {
            'epoch': self.curr_epoch,
            'total_iter': self.total_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'avg_kld': avg_kld,
            'avg_mlp': avg_mlp,
        }

        # Save the current checkpoint every self.save_freq epochs
        if self.curr_epoch % self.save_freq == 0:
            checkpoint_name = self.folder_name + f'checkpoints/epoch_{self.curr_epoch}.pt'
            print(f'Saving checkpoint at {checkpoint_name}')
            torch.save(checkpoint, checkpoint_name)

        # Save the best checkpoint based on KLD score
        if self.curr_epoch == 0 or avg_mlp < self.best_mlp:
            self.best_mlp = avg_mlp
            best_checkpoint_name = self.folder_name + 'checkpoints/best.pt'
            print('Saving best checkpoint')
            torch.save(checkpoint, best_checkpoint_name)

        self.curr_epoch += 1
