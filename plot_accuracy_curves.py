import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_training_history(filepath):
    """Load training history from JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_accuracy_curves():
    """Plot accuracy curves for all 4 models."""
    
    # Define model configurations
    models = [
        {
            'name': 'ResNet50',
            'history_file': 'training_history_resnet.json',
            'color': '#1f77b4',
            'linestyle': '-'
        },
        {
            'name': 'EfficientNet-B0',
            'history_file': 'training_history_efficientnet.json',
            'color': '#ff7f0e',
            'linestyle': '-'
        },
        {
            'name': 'VGG16-BN',
            'history_file': 'training_history_vgg16.json',
            'color': '#2ca02c',
            'linestyle': '-'
        },
        {
            'name': 'ViT-B/16',
            'history_file': 'training_history_vit.json',
            'color': '#d62728',
            'linestyle': '-'
        }
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Accuracy Curves - All Models', fontsize=16, fontweight='bold')
    
    # Plot 1: Base Training - Validation Accuracy
    ax1 = axes[0, 0]
    for model_config in models:
        history = load_training_history(model_config['history_file'])
        if history is None:
            continue
        
        if 'base_val_acc' in history and len(history['base_val_acc']) > 0:
            epochs = range(1, len(history['base_val_acc']) + 1)
            accuracies = [acc * 100 for acc in history['base_val_acc']]  # Convert to percentage
            ax1.plot(epochs, accuracies, 
                    label=f"{model_config['name']}", 
                    color=model_config['color'],
                    linestyle=model_config['linestyle'],
                    linewidth=2,
                    marker='o',
                    markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Phase 1: Base Training - Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Base Training - Training Accuracy
    ax2 = axes[0, 1]
    for model_config in models:
        history = load_training_history(model_config['history_file'])
        if history is None:
            continue
        
        if 'base_train_acc' in history and len(history['base_train_acc']) > 0:
            epochs = range(1, len(history['base_train_acc']) + 1)
            accuracies = [acc * 100 for acc in history['base_train_acc']]
            ax2.plot(epochs, accuracies, 
                    label=f"{model_config['name']}", 
                    color=model_config['color'],
                    linestyle=model_config['linestyle'],
                    linewidth=2,
                    marker='o',
                    markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax2.set_title('Phase 1: Base Training - Training Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # Plot 3: Fine-tuning - Validation Accuracy
    ax3 = axes[1, 0]
    for model_config in models:
        history = load_training_history(model_config['history_file'])
        if history is None:
            continue
        
        if 'tune_val_acc' in history and len(history['tune_val_acc']) > 0:
            epochs = range(1, len(history['tune_val_acc']) + 1)
            accuracies = [acc * 100 for acc in history['tune_val_acc']]
            ax3.plot(epochs, accuracies, 
                    label=f"{model_config['name']}", 
                    color=model_config['color'],
                    linestyle=model_config['linestyle'],
                    linewidth=2,
                    marker='o',
                    markersize=4)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax3.set_title('Phase 2: Fine-tuning - Validation Accuracy', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # Plot 4: Fine-tuning - Training Accuracy
    ax4 = axes[1, 1]
    for model_config in models:
        history = load_training_history(model_config['history_file'])
        if history is None:
            continue
        
        if 'tune_train_acc' in history and len(history['tune_train_acc']) > 0:
            epochs = range(1, len(history['tune_train_acc']) + 1)
            accuracies = [acc * 100 for acc in history['tune_train_acc']]
            ax4.plot(epochs, accuracies, 
                    label=f"{model_config['name']}", 
                    color=model_config['color'],
                    linestyle=model_config['linestyle'],
                    linewidth=2,
                    marker='o',
                    markersize=4)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax4.set_title('Phase 2: Fine-tuning - Training Accuracy', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'accuracy_curves_all_models.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy curves saved to {output_path}")
    
    # Also create a combined plot showing validation accuracy across both phases
    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for model_config in models:
        history = load_training_history(model_config['history_file'])
        if history is None:
            continue
        
        # Combine base and fine-tuning phases
        base_epochs = []
        base_accs = []
        tune_epochs = []
        tune_accs = []
        
        if 'base_val_acc' in history and len(history['base_val_acc']) > 0:
            base_epochs = list(range(1, len(history['base_val_acc']) + 1))
            base_accs = [acc * 100 for acc in history['base_val_acc']]
        
        if 'tune_val_acc' in history and len(history['tune_val_acc']) > 0:
            # Fine-tuning epochs continue from base training
            start_epoch = len(base_epochs) + 1 if base_epochs else 1
            tune_epochs = list(range(start_epoch, start_epoch + len(history['tune_val_acc'])))
            tune_accs = [acc * 100 for acc in history['tune_val_acc']]
        
        # Plot base phase
        if base_epochs:
            ax.plot(base_epochs, base_accs, 
                   color=model_config['color'],
                   linestyle='-',
                   linewidth=2,
                   marker='o',
                   markersize=4,
                   label=f"{model_config['name']} (Base)")
        
        # Plot fine-tuning phase
        if tune_epochs:
            ax.plot(tune_epochs, tune_accs, 
                   color=model_config['color'],
                   linestyle='--',
                   linewidth=2,
                   marker='s',
                   markersize=4,
                   label=f"{model_config['name']} (Fine-tuned)")
        
        # Add vertical line to separate phases
        if base_epochs and tune_epochs:
            ax.axvline(x=len(base_epochs) + 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy - All Models (Base Training + Fine-tuning)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add phase labels
    if base_epochs:
        ax.text(len(base_epochs) / 2, 95, 'Base Training', 
               ha='center', fontsize=11, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if tune_epochs:
        mid_tune = len(base_epochs) + len(tune_epochs) / 2 if base_epochs else len(tune_epochs) / 2
        ax.text(mid_tune, 95, 'Fine-tuning', 
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    output_path2 = 'accuracy_curves_combined.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Combined accuracy curve saved to {output_path2}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FINAL ACCURACY SUMMARY")
    print("="*60)
    
    for model_config in models:
        history = load_training_history(model_config['history_file'])
        if history is None:
            continue
        
        print(f"\n{model_config['name']}:")
        
        if 'base_val_acc' in history and len(history['base_val_acc']) > 0:
            final_base = history['base_val_acc'][-1] * 100
            print(f"  Base Training - Final Val Acc: {final_base:.2f}%")
        
        if 'tune_val_acc' in history and len(history['tune_val_acc']) > 0:
            final_tune = history['tune_val_acc'][-1] * 100
            improvement = final_tune - final_base if 'base_val_acc' in history and len(history['base_val_acc']) > 0 else 0
            print(f"  Fine-tuning - Final Val Acc: {final_tune:.2f}% (Improvement: {improvement:+.2f}%)")
    
    print("="*60)

if __name__ == '__main__':
    plot_accuracy_curves()

