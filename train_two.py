import os
import torch
import torch.nn as nn
import torch.optim as optim
import json

# Import your existing modules
from config import *
from data_loader import get_dataloaders
from model import setup_model
from train import train_model

def train_partners():
    # 1. Setup Data
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    
    # 2. Define ONLY the new partners (Skipping ResNet)
    partners = ['efficientnet_b0', 'vgg16_bn']
    
    print(f"Starting Training for Partners on {DEVICE}...")

    for arch in partners:
        print("\n" + "="*50)
        print(f"TRAINING PARTNER: {arch.upper()}")
        print("="*50)
        
        base_ckpt = f"best_{arch}.pth"
        if os.path.exists(base_ckpt):
            print(f"Skipping base training for {arch} (found {base_ckpt})")
            continue

        # --- A. Setup Model ---
        # Note: Ensure your model.py has the factory logic I gave you in the previous step!
        model = setup_model(num_classes, DEVICE, arch=arch, freeze_base=True)
        
        # --- B. Define Optimizer (Targeting correct layers) ---
        if arch == 'efficientnet_b0':
            # EfficientNet classifier is a Sequential, get all parameters
            params = model.classifier.parameters()
        elif arch == 'vgg16_bn':
            # VGG16 classifier[6] is now a Sequential, get its parameters
            params = model.classifier[6].parameters()
        else:
            # Fallback (shouldn't happen)
            params = model.parameters()
            
        optimizer = optim.Adam(params, lr=BASE_LR)
        criterion = nn.CrossEntropyLoss()

        # --- C. Train ---
        save_path = base_ckpt
        
        # We use your existing train_model function
        trained_model, history = train_model(
            model, dataloaders, criterion, optimizer, 
            num_epochs=NUM_EPOCHS_BASE, 
            checkpoint_path=save_path, 
            image_datasets=image_datasets
        )
        
        print(f"Finished {arch}. Saved to {save_path}")

    print("\nPartner training complete! You now have a full committee.")

def fine_tune_partners():
    # 1. Setup Data
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    
    # 2. Define partners
    partners = ['efficientnet_b0', 'vgg16_bn']
    
    print(f"Starting Fine-Tuning for Partners on {DEVICE}...")

    for arch in partners:
        base_ckpt = f"best_{arch}.pth"
        tuned_ckpt = f"best_{arch}_tuned.pth"

        if not os.path.exists(base_ckpt):
            print(f"Skipping fine-tuning for {arch}: base checkpoint not found ({base_ckpt})")
            continue

        print("\n" + "="*50)
        print(f"FINE-TUNING PARTNER: {arch.upper()}")
        print("="*50)
        
        # --- A. Setup Model (unfrozen) and load base weights ---
        model = setup_model(num_classes, DEVICE, arch=arch, freeze_base=False)
        model.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))

        # --- B. Optimizer on all params (since unfrozen) ---
        optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        criterion = nn.CrossEntropyLoss()

        # --- C. Train (phase 2) ---
        trained_model, history = train_model(
            model, dataloaders, criterion, optimizer, 
            num_epochs=NUM_EPOCHS_TUNE, 
            checkpoint_path=tuned_ckpt, 
            image_datasets=image_datasets
        )
        
        print(f"Finished fine-tuning {arch}. Saved to {tuned_ckpt}")

    print("\nPartner fine-tuning complete!")

if __name__ == '__main__':
    # Base training (skips if checkpoints already exist)
    train_partners()
    # Fine-tuning from saved base checkpoints (second layer)
    fine_tune_partners()