"""
Training script for Multi-Layer Perceptron Multiclass Classification (5 classes).
Matches train_svm.py structure with resampling, PCA, and K-Fold CV.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from training.classification.multi_layer_perceptron import MLPFromScratch

# Configuration
RANDOM_STATE = 42
LEARNING_RATE = 0.006
N_ITERATIONS = 1000
HIDDEN_LAYERS = [32]


class OneVsRestMLP:
    """One-vs-Rest wrapper for multiclass MLP classification."""
    
    def __init__(self, input_size, learning_rate=0.01, num_iterations=500, 
                 hidden_layers=[32], use_cuda=False):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.hidden_layers = hidden_layers
        self.use_cuda = use_cuda
        self.classifiers_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        
        for cls in self.classes_:
            print(f"\nTraining classifier for class {cls}...")
            # Create layer sizes: input -> hidden layers -> output (1 for binary)
            layer_sizes = [self.input_size] + self.hidden_layers + [1]
            
            mlp = MLPFromScratch(
                layer_sizes=layer_sizes,
                learning_rate=self.learning_rate,
                num_iterations=self.num_iterations,
                use_cuda=self.use_cuda
            )
            binary_y = np.where(y == cls, 1, 0)
            mlp.fit(X, binary_y)
            self.classifiers_.append(mlp)
        return self
    
    def predict_proba(self, X):
        # Get raw probabilities from each binary classifier
        raw_probs = []
        for clf in self.classifiers_:
            proba = clf.predict_proba(X)
            # Ensure it's a NumPy array and flatten
            if hasattr(proba, 'get'):  # CuPy array
                proba = proba.get()
            proba = np.asarray(proba).flatten()
            raw_probs.append(proba)
        
        scores = np.column_stack(raw_probs)
        # Convert scores to probabilities using softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def main(data_path):
    """Main training pipeline for 5-class classification matching train_svm.py."""
    # Load data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    if df['Booking Status'].dtype == 'object':
        le = LabelEncoder()
        df['Booking Status'] = le.fit_transform(df['Booking Status'])
    
    X = df.drop('Booking Status', axis=1)
    y = df['Booking Status']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Downsample to median (matching SVM)
    class_counts = Counter(y_train)
    median_count = int(np.median(list(class_counts.values())))
    downsample_strategy = {cls: min(count, median_count) for cls, count in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=downsample_strategy, random_state=42)
    X_under, y_under = rus.fit_resample(X_train, y_train)
    target_3 = median_count

    ros = RandomOverSampler(sampling_strategy={3: target_3}, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_under, y_under)

    print(f"After resampling: {X_resampled.shape}")
    print(f"Class distribution: {Counter(y_resampled)}")

    # Scale first, then PCA (correct order, no data leakage)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # PCA after scaling - fit on train only, transform both
    n_components = min(40, X_resampled_scaled.shape[1], len(X_resampled_scaled)-1)
    pca = PCA(n_components=n_components, random_state=42)
    X_resampled_pca = pca.fit_transform(X_resampled_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    train_scaled = X_resampled_pca
    test_scaled = X_test_pca

    print(f"PCA reduced to {train_scaled.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.2f} variance")

    y_resampled_array = y_resampled.values if hasattr(y_resampled, 'values') else y_resampled
    
    # Train final model with K-Fold
    n_splits = 11
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    n_classes = len(np.unique(y_resampled_array))
    oof = np.zeros((len(train_scaled), n_classes))
    preds_test = np.zeros((len(test_scaled), n_classes))
    
    print(f"\nTraining MLP with {n_splits}-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_scaled, y_resampled_array)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        mlp = OneVsRestMLP(
            input_size=train_scaled.shape[1],
            learning_rate=LEARNING_RATE,
            num_iterations=N_ITERATIONS,
            hidden_layers=HIDDEN_LAYERS,
            use_cuda=True  # Enable CUDA acceleration
        )
        mlp.fit(train_scaled[train_idx], y_resampled_array[train_idx])
        
        oof[val_idx] = mlp.predict_proba(train_scaled[val_idx])
        preds_test += mlp.predict_proba(test_scaled) / n_splits
        print(f"\nFold {fold+1} completed")
    
    train_pred = np.argmax(oof, axis=1)
    test_pred = np.argmax(preds_test, axis=1)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test F1 (Macro): {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(f"Test F1 (Weighted): {f1_score(y_test, test_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix (raw counts):")
    print(conf_mat)

    conf_mat_norm = confusion_matrix(y_test, test_pred, normalize='true')
    print("\nConfusion Matrix (normalized):")
    print(np.round(conf_mat_norm, 3))

    # Create output directory if it doesn't exist
    os.makedirs('output/plots', exist_ok=True)

    # Plot confusion matrix
    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp1.plot(ax=ax[0], cmap='Blues', colorbar=False)
    ax[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight='bold')

    disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm)
    disp2.plot(ax=ax[1], cmap='Blues', colorbar=False)
    ax[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    cm_path = 'output/plots/mlp_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {cm_path}")
    plt.close()

    precision, recall, f1, support = precision_recall_fscore_support(y_test, test_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(precision))
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics (MLP)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in range(len(precision))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    metrics_path = 'output/plots/mlp_class_metrics.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class metrics plot to {metrics_path}")
    plt.close()

    # Plot prediction distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # True distribution
    true_counts = np.bincount(y_test)
    pred_counts = np.bincount(test_pred, minlength=len(true_counts))

    x = np.arange(len(true_counts))
    ax1.bar(x, true_counts, alpha=0.7, label='True', color='green')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Predicted distribution
    ax2.bar(x, pred_counts, alpha=0.7, label='Predicted', color='orange')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    dist_path = 'output/plots/mlp_class_distribution.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class distribution plot to {dist_path}")
    plt.close()

    print(f"\n{'='*60}")
    print("All plots saved to output/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MLP classifier (5-class)')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')

    args = parser.parse_args()
    main(args.data)
