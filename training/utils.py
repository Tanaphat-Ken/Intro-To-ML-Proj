"""
Utility functions for model training and evaluation.
"""
import pickle
import os


def save_model_weights(model, filepath):
    """
    Save model weights to file.

    Parameters:
    -----------
    model : object
        Model instance with attributes to save
    filepath : str
        Path to save the weights
    """
    # Get all non-private attributes
    weights = {k: v for k, v in model.__dict__.items() if not k.startswith('_')}

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath):
    """
    Load model weights from file.

    Parameters:
    -----------
    model : object
        Model instance to load weights into
    filepath : str
        Path to load the weights from
    """
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)

    for k, v in weights.items():
        setattr(model, k, v)

    print(f"Model weights loaded from {filepath}")
