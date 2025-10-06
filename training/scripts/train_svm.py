import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

from training.classification.support_vector_machine import KernelSVMClassifier
class OneVsRestKernelSVM:    
    def __init__(self, kernel='rbf', C=1.0, degree=3, gamma=None, learning_rate=0.001, n_iterations=1000):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.classifiers_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        
        for cls in self.classes_:
            svm = KernelSVMClassifier(
                kernel=self.kernel,
                C=self.C,
                degree=self.degree,
                gamma=self.gamma,
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations
            )
            binary_y = np.where(y == cls, 1, 0)
            svm.fit(X, binary_y)
            self.classifiers_.append(svm)
        return self
    
    def predict_proba(self, X):
        scores = np.column_stack([clf.decision_function(X) for clf in self.classifiers_])
        # Convert scores to probabilities using softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

def main(data_path):
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
    
    # Downsample to median
    class_counts = Counter(y_train)
    median_count = int(np.median(list(class_counts.values())))
    downsample_strategy = {cls: min(count, median_count) for cls, count in class_counts.items()}
    
    rus = RandomUnderSampler(sampling_strategy=downsample_strategy, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"After downsampling: {X_resampled.shape}")
    print(f"Class distribution: {Counter(y_resampled)}")

    # Scale first, then PCA (correct order, no data leakage)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # PCA after scaling - fit on train only, transform both
    n_components = min(30, X_resampled_scaled.shape[1], len(X_resampled_scaled)-1)
    pca = PCA(n_components=n_components, random_state=42)
    X_resampled_pca = pca.fit_transform(X_resampled_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    train_scaled = X_resampled_pca
    test_scaled = X_test_pca

    print(f"PCA reduced to {train_scaled.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.2f} variance")

    y_resampled_array = y_resampled.values if hasattr(y_resampled, 'values') else y_resampled
    
    # Use pre-tuned parameters (based on NuSVC results)
    # Original tuning showed: kernel='poly', degree=4, gamma='auto', nu=0.4, coef0=0.08
    print("\nUsing pre-tuned parameters (kernel='poly', degree=4)...")
    best_params = {'kernel': 'poly', 'C': 1.0, 'degree': 4}
    print(f"Parameters: {best_params}")

    # # Grid search for best kernel params (commented out to save time)
    # print("\nSearching for best kernel parameters...")
    # param_grid = {
    #     'kernel': ['rbf', 'poly'],
    #     'C': [0.1, 1.0, 10.0],
    #     'degree': [2, 3, 4],  # for poly
    # }
    #
    # best_score = -np.inf
    # best_params = None
    #
    # for kernel in param_grid['kernel']:
    #     for C in param_grid['C']:
    #         if kernel == 'poly':
    #             degrees = param_grid['degree']
    #         else:
    #             degrees = [3]  # dummy for rbf
    #
    #         for degree in degrees:
    #             # Quick CV test with 3 folds
    #             skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    #             scores = []
    #
    #             for train_idx, val_idx in skf.split(train_scaled, y_resampled_array):
    #                 svm = OneVsRestKernelSVM(
    #                     kernel=kernel,
    #                     C=C,
    #                     degree=degree,
    #                     learning_rate=0.01,
    #                     n_iterations=500
    #                 )
    #                 svm.fit(train_scaled[train_idx], y_resampled_array[train_idx])
    #                 pred = svm.predict(train_scaled[val_idx])
    #                 scores.append(accuracy_score(y_resampled_array[val_idx], pred))
    #
    #             mean_score = np.mean(scores)
    #             if mean_score > best_score:
    #                 best_score = mean_score
    #                 best_params = {'kernel': kernel, 'C': C, 'degree': degree}
    #                 print(f"kernel={kernel}, C={C}, degree={degree}: {mean_score:.4f}")
    #
    # print(f"\nBest params: {best_params}, CV Score: {best_score:.4f}")
    
    # Train final model with K-Fold
    n_splits = 11
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    n_classes = len(np.unique(y_resampled_array))
    oof = np.zeros((len(train_scaled), n_classes))
    preds_test = np.zeros((len(test_scaled), n_classes))
    
    print(f"\nTraining with {n_splits}-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_scaled, y_resampled_array)):
        print(f"Fold {fold+1}/{n_splits}...", end=" ")
        
        svm = OneVsRestKernelSVM(
            kernel=best_params['kernel'],
            C=best_params['C'],
            degree=best_params['degree'],
            learning_rate=0.01,
            n_iterations=1000
        )
        svm.fit(train_scaled[train_idx], y_resampled_array[train_idx])
        
        oof[val_idx] = svm.predict_proba(train_scaled[val_idx])
        preds_test += svm.predict_proba(test_scaled) / n_splits
        print("Done")
    
    train_pred = np.argmax(oof, axis=1)
    test_pred = np.argmax(preds_test, axis=1)
    
    print(f"\nTest Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test F1 (Macro): {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(f"Test F1 (Weighted): {f1_score(y_test, test_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train SVM classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')

    args = parser.parse_args()
    main(args.data)