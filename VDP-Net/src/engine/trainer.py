import sys, os
import numpy as np
import torch
from PIL import Image
import tqdm
import time
import torch.nn.functional as F

from src.engine import utils

class BaseTrainer():
    def __init__(self, args, model, train_set, val_set, writer):
        self.args = args
        self.folder_name = args.output_folder
        self.writer = writer
        self.display_freq = args.display_freq
        self.log_freq = args.log_freq
        self.save_freq = args.save_freq
        
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.epochs = args.epochs
        self.start_epoch = 1
        self.curr_epoch = 1
        self.total_iter = 0

        self.model = model
        self.device = args.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        
        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers)
        
        self.val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers)
        
    def load_from_checkpoint(self, checkpoint):
        raise NotImplementedError("Load from checkpoint method must be implemented in subclass.")

        
    def train(self):
        raise NotImplementedError("Train method must be implemented in subclass.")
    
    def validate(self):
        raise NotImplementedError("Validate method must be implemented in subclass.")
    
    def run(self):
        # Training loop
        for epoch in range(self.start_epoch, self.epochs+1):
            print(f"\n[Epoch {epoch}/{self.epochs}]")
            start_time = time.time()
            self.train()
            torch.cuda.empty_cache()
            self.validate()
            torch.cuda.empty_cache()
            end_time = time.time()
            print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds.")