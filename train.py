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
    # --- Phase 1: Base Training ---
    print("\n" + "="*40 + "\nSTARTING PHASE 1: BASE TRAINING\n" + "="*40)
    
    model = setup_model(NUM_CLASSES, DEVICE, freeze_base=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_base = optim.Adam(model.fc.parameters(), lr=BASE_LR)

    # Function call parameters on one line
    final_model, history_base = train_model(model, dataloaders, criterion, optimizer_base, NUM_EPOCHS_BASE, CHECKPOINT_PATH_BASE, image_datasets)
    print(f"Base training complete. Model saved to: {CHECKPOINT_PATH_BASE}")

    # --- Phase 2: Fine-Tuning ---
    print("\n" + "="*40 + "\nSTARTING PHASE 2: FINE-TUNING\n" + "="*40)
    
    final_model = setup_model(NUM_CLASSES, DEVICE, freeze_base=False)
    final_model.load_state_dict(torch.load(CHECKPOINT_PATH_BASE))
    optimizer_tune = optim.Adam(final_model.parameters(), lr=FINE_TUNE_LR)

    # Function call parameters on one line
    final_model_tuned, history_fine_tune = train_model(final_model, dataloaders, criterion, optimizer_tune, NUM_EPOCHS_TUNE, CHECKPOINT_PATH_TUNE, image_datasets)
    print(f"Fine-tuning complete. Model saved to: {CHECKPOINT_PATH_TUNE}")

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