import torch
from PIL import Image
import sys
import os

from config import DEVICE, CHECKPOINT_PATH_TUNE, IMAGENET_MEAN, IMAGENET_STD
from model import setup_model
from data_loader import get_dataloaders, get_data_transforms

def load_inference_model(checkpoint_path, num_classes, device):
    model = setup_model(num_classes, device, freeze_base=False)
    
    # Load the trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() 
    return model

def predict_single_image(image_path, model, transform, class_names, device):
    image = Image.open(image_path).convert('RGB')
    
    # Apply the validation transforms (Resize -> CenterCrop -> ToTensor -> Normalize)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score

if __name__ == '__main__':
    # 1. Setup: Get class names and transforms
    # We load dataloaders just to extract the consistent class names list
    _, _, NUM_CLASSES, class_names = get_dataloaders()
    transforms_dict = get_data_transforms()
    val_transform = transforms_dict['val']

    # 2. Load the best trained model
    print(f"Loading model from {CHECKPOINT_PATH_TUNE}...")
    model = load_inference_model(CHECKPOINT_PATH_TUNE, NUM_CLASSES, DEVICE)

    # 3. Define the image to test (Change this path to test different images!)
    # You can pass the image path as a command line argument, or default to a specific file
    test_image_path = sys.argv[1] 
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        
    else:
        './data/val/Baroque/test_example.jpg'

    if not os.path.exists(test_image_path):
        print(f"Error: Image not found at {test_image_path}")
    else:
        # 4. Run Prediction
        # Function parameters all on one line
        era, conf = predict_single_image(test_image_path, model, val_transform, class_names, DEVICE)
        
        print("\n" + "="*30)
        print(f"Prediction: {era}")
        print(f"Confidence: {conf*100:.2f}%")
        print("="*30 + "\n")