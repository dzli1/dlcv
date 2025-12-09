import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
import json

# Import from local modules
from config import *
from data_loader import get_dataloaders
from model import setup_model

# Load Data
dataloaders, image_datasets, NUM_CLASSES, class_names = get_dataloaders()

def train_model(model, dataloaders, criterion, optimizer, num_epochs, checkpoint_path, image_datasets):
    best_model_wts = copy.deepcopy(model.state_dict()); best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0; running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()}ing"):
                inputs = inputs.to(DEVICE); labels = labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved! New best validation accuracy: {best_acc:.4f}")
                    
    model.load_state_dict(best_model_wts)
    return model, history

if __name__ == '__main__':
    ARCH = 'vit_b_16'
    BASE_CHECKPOINT = 'best_vit_b_16_base.pth'
    TUNED_CHECKPOINT = 'best_vit_b_16_tuned.pth'
    HISTORY_FILE = 'training_history_vit.json'
    
    # --- Phase 1: Base Training ---
    print("\n" + "="*50 + f"\nTRAINING {ARCH.upper()}: PHASE 1 - BASE TRAINING\n" + "="*50)
    
    model = setup_model(NUM_CLASSES, DEVICE, arch=ARCH, freeze_base=True)
    criterion = nn.CrossEntropyLoss()
    # ViT head is a Sequential, get all parameters
    optimizer_base = optim.Adam(model.heads.head.parameters(), lr=BASE_LR)

    final_model, history_base = train_model(model, dataloaders, criterion, optimizer_base, NUM_EPOCHS_BASE, BASE_CHECKPOINT, image_datasets)
    print(f"Base training complete. Model saved to: {BASE_CHECKPOINT}")

    # --- Phase 2: Fine-Tuning ---
    print("\n" + "="*50 + f"\nTRAINING {ARCH.upper()}: PHASE 2 - FINE-TUNING\n" + "="*50)
    
    final_model = setup_model(NUM_CLASSES, DEVICE, arch=ARCH, freeze_base=False)
    final_model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=DEVICE))
    optimizer_tune = optim.Adam(final_model.parameters(), lr=FINE_TUNE_LR)

    final_model_tuned, history_fine_tune = train_model(final_model, dataloaders, criterion, optimizer_tune, NUM_EPOCHS_TUNE, TUNED_CHECKPOINT, image_datasets)
    print(f"Fine-tuning complete. Model saved to: {TUNED_CHECKPOINT}")

    # --- Save History ---
    full_history = {
        'base_train_loss': history_base['train_loss'], 'base_val_loss': history_base['val_loss'],
        'base_train_acc': history_base['train_acc'], 'base_val_acc': history_base['val_acc'],
        'tune_train_loss': history_fine_tune['train_loss'], 'tune_val_loss': history_fine_tune['val_loss'],
        'tune_train_acc': history_fine_tune['train_acc'], 'tune_val_acc': history_fine_tune['val_acc']
    }

    with open(HISTORY_FILE, 'w') as f:
        json.dump(full_history, f, indent=4)
    print(f"\nTraining history saved to {HISTORY_FILE}")

