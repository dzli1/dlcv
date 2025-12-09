import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
import os

from config import *
from model import setup_model
from data_loader import get_dataloaders, get_data_transforms

def load_committee(num_classes, device):
    """
    Loads:
    1. Your ALREADY TRAINED ResNet (The Captain)
    2. The NEW EfficientNet (Partner 1)
    3. The NEW VGG16 (Partner 2)
    """
    models_dict = {}
    
    # --- MAP ARCHITECTURES TO FILENAMES ---
    # Update 'resnet_path' to match whatever you named your first model!
    committee_config = {
        'resnet50': 'final_best_tuned_model.pth',   # <-- YOUR EXISTING MODEL
        'efficientnet_b0': 'best_efficientnet_b0.pth', # <-- New model
        'vgg16_bn': 'best_vgg16_bn.pth'              # <-- New model
    }
    
    for arch, path in committee_config.items():
        if os.path.exists(path):
            print(f"Loading {arch} from {path}...")
            # Create the skeleton
            m = setup_model(num_classes, device, arch=arch, freeze_base=False)
            # Load the weights
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models_dict[arch] = m
        else:
            print(f"⚠️ Warning: {path} not found. {arch} will be excluded from the vote.")
            
    return models_dict

def predict_consensus(image_path, models_dict, transform, class_names, device):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    total_probs = None
    
    print("\n--- Individual Votes ---")
    with torch.no_grad():
        for name, model in models_dict.items():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Print individual guess
            conf, idx = torch.max(probs, 1)
            print(f"{name}: {class_names[idx.item()]} ({conf.item()*100:.1f}%)")
            
            # Add to consensus
            if total_probs is None:
                total_probs = probs
            else:
                total_probs += probs
    
    # Average the votes
    if len(models_dict) > 0:
        avg_probs = total_probs / len(models_dict)
        final_conf, final_idx = torch.max(avg_probs, 1)
        winner = class_names[final_idx.item()]
        return winner, final_conf.item()
    else:
        return "Error: No models loaded", 0.0

if __name__ == '__main__':
    # Setup
    _, _, num_classes, class_names = get_dataloaders()
    val_transform = get_data_transforms()['val']
    
    # Load Committee
    committee = load_committee(num_classes, DEVICE)
    
    if not committee:
        print("No models found! Check filenames.")
        sys.exit(1)
    
    # Predict
    test_image = sys.argv[1] if len(sys.argv) > 1 else './data/val/Baroque/test_example.jpg'
    
    if os.path.exists(test_image):
        print(f"\nConsensus for: {test_image}")
        winner, conf = predict_consensus(test_image, committee, val_transform, class_names, DEVICE)
        print("="*30)
        print(f"FINAL CONSENSUS: {winner}")
        print(f"Confidence: {conf*100:.2f}%")
        print("="*30)
    else:
        print(f"Error: Image not found at {test_image}")