import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history():
    # Load training history
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Base Training Loss
    ax1 = axes[0, 0]
    epochs_base = range(1, len(history['base_train_loss']) + 1)
    ax1.plot(epochs_base, history['base_train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_base, history['base_val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Phase 1: Base Training (Frozen Backbone)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Base Training Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs_base, history['base_train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_base, history['base_val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Phase 1: Base Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fine-tuning Loss
    ax3 = axes[1, 0]
    epochs_tune = range(1, len(history['tune_train_loss']) + 1)
    ax3.plot(epochs_tune, history['tune_train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax3.plot(epochs_tune, history['tune_val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Phase 2: Fine-tuning (Unfrozen)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fine-tuning Accuracy
    ax4 = axes[1, 1]
    ax4.plot(epochs_tune, history['tune_train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax4.plot(epochs_tune, history['tune_val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Phase 2: Fine-tuning Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to training_curves.png")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"\nPhase 1 (Base Training):")
    print(f"  Final Train Loss: {history['base_train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['base_val_loss'][-1]:.4f}")
    print(f"  Final Train Acc: {history['base_train_acc'][-1]*100:.2f}%")
    print(f"  Final Val Acc: {history['base_val_acc'][-1]*100:.2f}%")
    
    print(f"\nPhase 2 (Fine-tuning):")
    print(f"  Final Train Loss: {history['tune_train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['tune_val_loss'][-1]:.4f}")
    print(f"  Final Train Acc: {history['tune_train_acc'][-1]*100:.2f}%")
    print(f"  Final Val Acc: {history['tune_val_acc'][-1]*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    plot_training_history()

