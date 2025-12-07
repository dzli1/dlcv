"""
Interpretability and visualization utilities:
- Grad-CAM for visualizing model attention
- Embedding extraction and t-SNE/UMAP visualization
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

class GradCAM:
    """Grad-CAM implementation for ResNet models."""
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer to compute gradients for (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, class_idx=None, task='movement'):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Class index to generate CAM for. If None, uses predicted class.
            task: 'movement' or 'artist' for multi-task models
        
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle new model format (dict with movement_logits/artist_logits)
        if isinstance(output, dict):
            logits = output[f'{task}_logits']
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
        else:
            logits = output
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        logits[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:])  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()

def visualize_gradcam(model, image_path, target_layer, class_names, 
                     true_label=None, save_path=None, device='cpu', task='movement', image_size=224):
    """
    Visualize Grad-CAM for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        target_layer: Layer to visualize (e.g., model.backbone.layer4 or model.backbone.stages[-1])
        class_names: Dict mapping class index to name
        true_label: True label index (optional)
        save_path: Path to save visualization
        device: Device to run on
        task: 'movement' or 'artist' for multi-task models
        image_size: Image size for preprocessing
    """
    # Load and preprocess image
    from torchvision import transforms
    from config import IMAGENET_MEAN, IMAGENET_STD
    
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            logits = output[f'{task}_logits']
        else:
            logits = output
        probs = F.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(img_tensor, class_idx=pred_idx, task=task)
    
    # Resize CAM to match image
    cam_resized = cv2.resize(cam, (image_size, image_size))
    cam_resized = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # Convert image to numpy
    img_np = np.array(img.resize((image_size, image_size)))
    
    # Overlay
    overlay = 0.4 * img_np + 0.6 * heatmap[:, :, ::-1]  # BGR to RGB
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap[:, :, ::-1])
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    pred_name = class_names.get(pred_idx, f"Class {pred_idx}")
    title = f'Predicted: {pred_name} ({confidence:.2f})'
    if true_label is not None:
        true_name = class_names.get(true_label, f"Class {true_label}")
        title += f'\nTrue: {true_name}'
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return pred_idx, confidence, cam

def extract_embeddings(model, dataloader, device='cpu', task='movement'):
    """
    Extract embeddings from model features.
    
    Args:
        model: Trained model (MultiTaskClassifier)
        dataloader: DataLoader (returns tuples: images, movements, artists, record_ids)
        device: Device to run on
        task: 'movement' or 'artist' to determine which labels to use
    
    Returns:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        record_ids: List of record IDs
    """
    model.eval()
    embeddings = []
    labels = []
    record_ids = []
    
    with torch.no_grad():
        for images, movements, artists, record_id in dataloader:
            images = images.to(device, non_blocking=True)
            
            # Forward pass to get features
            outputs = model(images)
            features = outputs["features"]
            
            # Use appropriate labels
            if task == 'movement':
                task_labels = movements
            else:
                task_labels = artists
            
            embeddings.append(features.cpu().numpy())
            labels.append(task_labels.numpy())
            record_ids.extend(record_id)
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    return embeddings, labels, record_ids

def visualize_embeddings(embeddings, labels, class_names, method='tsne', 
                        save_path=None, n_components=2, perplexity=30):
    """
    Visualize embeddings using t-SNE or UMAP.
    
    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array of class indices
        class_names: List of class names
        method: 'tsne' or 'umap'
        save_path: Path to save visualization
        n_components: Number of dimensions for reduction (2 or 3)
        perplexity: Perplexity for t-SNE
    """
    print(f"Reducing {embeddings.shape[0]} samples from {embeddings.shape[1]}D to {n_components}D using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                      random_state=42, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        if not HAS_UMAP:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    
    if n_components == 2:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab20', alpha=0.6, s=20)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
    else:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], embeddings_2d[:, 2],
                           c=labels, cmap='tab20', alpha=0.6, s=20)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')
    
    # Add legend
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=plt.cm.tab20(i), markersize=10)
               for i in range(len(unique_labels))]
    plt.legend(handles, [class_names[int(l)] for l in unique_labels], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return embeddings_2d

