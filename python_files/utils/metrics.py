
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import json


def calculate_metrics(y_true, y_pred, class_names=None):

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Generate classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix', normalize=True):
  
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'

    # Create figure
    n_classes = len(class_names)
    figsize = max(10, n_classes * 0.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Plot heatmap
    sns.heatmap(
        cm, annot=(n_classes <= 20), fmt=fmt, cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)

    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path, title='Training History'):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy Curves', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_json(metrics, save_path):

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    metrics_serializable = convert_types(metrics)

    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
