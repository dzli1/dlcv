"""
Script to run interpretability visualizations after training.
"""
import torch
import argparse
from pathlib import Path
from config import DEVICE, IMAGE_SIZE_CNN, CHECKPOINT_DIR, EMBEDDING_DIR, CONFUSION_DIR
from data_loader import get_dataloaders, load_label_mappings
from model import MultiTaskClassifier
from visualize import visualize_gradcam, extract_embeddings, visualize_embeddings

def get_target_layer(model, arch):
    """Get the target layer for Grad-CAM based on architecture."""
    try:
        if arch == "resnet50":
            # ResNet50: use the last conv layer in layer4
            if hasattr(model.backbone, 'layer4'):
                return model.backbone.layer4[-1].conv3
            elif hasattr(model.backbone, 'blocks'):
                # Some ResNet variants
                return model.backbone.blocks[-1]
        elif "convnext" in arch.lower():
            # ConvNeXt: use last block in last stage
            if hasattr(model.backbone, 'stages'):
                return model.backbone.stages[-1].blocks[-1]
            elif hasattr(model.backbone, 'blocks'):
                return model.backbone.blocks[-1]
        elif "efficientnet" in arch.lower():
            # EfficientNet: use last block
            if hasattr(model.backbone, 'blocks'):
                return model.backbone.blocks[-1]
        elif "vit" in arch.lower() or "vision_transformer" in arch.lower():
            # Vision Transformer: use last attention block
            if hasattr(model.backbone, 'blocks'):
                return model.backbone.blocks[-1].norm1
            elif hasattr(model.backbone, 'layers'):
                return model.backbone.layers[-1].norm1
        
        # Fallback: try common patterns
        if hasattr(model.backbone, 'layer4'):
            return model.backbone.layer4[-1]
        elif hasattr(model.backbone, 'stages'):
            return model.backbone.stages[-1].blocks[-1]
        elif hasattr(model.backbone, 'blocks'):
            return model.backbone.blocks[-1]
        else:
            # Last resort: return the backbone itself (will work but less precise)
            return model.backbone
    except Exception as e:
        print(f"Warning: Error finding target layer: {e}")
        # Return backbone as fallback
        return model.backbone

def main():
    parser = argparse.ArgumentParser(description='Run model visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--mode', type=str, choices=['gradcam', 'embeddings', 'both'],
                       default='both', help='Visualization mode')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Path to single image for Grad-CAM')
    parser.add_argument('--task', type=str, choices=['movement', 'artist'],
                       default='movement', help='Task to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize from validation set')
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    arch = config.get('arch', 'convnext_tiny')
    
    # Load data
    print("Loading data...")
    dataloaders, datasets, dataset_info = get_dataloaders()
    movement_map = dataset_info["movement_map"]
    artist_map = dataset_info["artist_map"]
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = MultiTaskClassifier(
        arch=arch,
        num_movements=dataset_info["num_movements"],
        num_artists=dataset_info["num_artists"],
        dropout=config.get('dropout', 0.3),
        pretrained=False,  # Already loaded from checkpoint
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Get target layer for Grad-CAM
    try:
        target_layer = get_target_layer(model, arch)
        print(f"Using target layer: {target_layer}")
    except Exception as e:
        print(f"Warning: Could not determine target layer: {e}")
        target_layer = None
    
    # Grad-CAM visualization
    if args.mode in ['gradcam', 'both'] and target_layer is not None:
        print("\n" + "="*40)
        print("Running Grad-CAM visualization...")
        print("="*40)
        
        class_names = movement_map if args.task == 'movement' else artist_map
        
        if args.image_path:
            # Single image
            visualize_gradcam(
                model, args.image_path, target_layer, class_names,
                save_path=EMBEDDING_DIR / f'gradcam_{args.task}_single.png',
                device=DEVICE,
                task=args.task,
                image_size=IMAGE_SIZE_CNN
            )
        else:
            # Sample images from validation set
            print(f"Sampling {args.num_samples} images from validation set...")
            val_dataset = datasets['val']
            for i in range(min(args.num_samples, len(val_dataset))):
                sample = val_dataset[i]
                # New format: (image, movement, artist, record_id)
                image, movement, artist, record_id = sample
                
                # Save image temporarily to visualize
                from PIL import Image
                import tempfile
                import os
                
                # Convert tensor back to image for visualization
                # We need the actual image path from the dataset
                img_path = val_dataset.image_root / val_dataset.df.iloc[i]["image_path"]
                
                true_label = movement.item() if args.task == 'movement' else artist.item()
                
                try:
                    visualize_gradcam(
                        model, str(img_path), target_layer, class_names,
                        true_label=true_label,
                        save_path=EMBEDDING_DIR / f'gradcam_{args.task}_sample_{i}_{record_id}.png',
                        device=DEVICE,
                        task=args.task,
                        image_size=IMAGE_SIZE_CNN
                    )
                except Exception as e:
                    print(f"Error visualizing sample {i}: {e}")
                    continue
    
    # Embedding visualization
    if args.mode in ['embeddings', 'both']:
        print("\n" + "="*40)
        print("Extracting embeddings and creating t-SNE/UMAP visualization...")
        print("="*40)
        
        # Extract embeddings from validation set
        embeddings, labels, record_ids = extract_embeddings(
            model, dataloaders['val'], device=DEVICE, task=args.task
        )
        
        class_names = movement_map if args.task == 'movement' else artist_map
        
        # t-SNE
        print("Creating t-SNE visualization...")
        try:
            visualize_embeddings(
                embeddings, labels, class_names, method='tsne',
                save_path=EMBEDDING_DIR / f'embeddings_{args.task}_tsne.png'
            )
        except Exception as e:
            print(f"Error creating t-SNE: {e}")
        
        # UMAP (if available)
        try:
            print("Creating UMAP visualization...")
            visualize_embeddings(
                embeddings, labels, class_names, method='umap',
                save_path=EMBEDDING_DIR / f'embeddings_{args.task}_umap.png'
            )
        except ImportError:
            print("UMAP not available, skipping...")
        except Exception as e:
            print(f"Error creating UMAP: {e}")
    
    print(f"\nVisualizations complete! Check the {EMBEDDING_DIR} directory.")

if __name__ == '__main__':
    main()
