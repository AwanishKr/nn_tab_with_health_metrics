import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class Custom_Dataset_normal(Dataset):
    def __init__(self, x, y, feature_columns=None):
        """
        Custom PyTorch Dataset for standard tabular data training.
        
        Args:
            x (pd.DataFrame or numpy.ndarray): Feature data
            y (pd.Series/DataFrame or numpy.ndarray): Target data
            feature_columns (list, optional): List of feature column names (for DataFrame input)
        """
        if isinstance(x, pd.DataFrame):
            # DataFrame input - extract features
            if feature_columns is None:
                feature_columns = x.columns.tolist()
            
            # Extract feature data
            self.x_data = x[feature_columns].values.astype(np.float32)
        else:
            # Numpy array input (backward compatibility)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            self.x_data = x.astype(np.float32)
            
        # Handle target data
        if hasattr(y, 'values'):  # pandas Series/DataFrame
            self.y_data = y.values.astype(np.int64)
        else:  # numpy array
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            self.y_data = y.astype(np.int64)
 
    def __len__(self):
        return len(self.x_data)
 
    def __getitem__(self, idx):
        x_sample = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y_sample = torch.tensor(self.y_data[idx], dtype=torch.long)
        
        return x_sample, y_sample



class Custom_Dataset_CRL(Dataset):
    def __init__(self, x, y, feature_columns=None):
        """
        Custom PyTorch Dataset for confidence-aware learning that uses dataset index (idx) as identifier.
        
        Args:
            x (pd.DataFrame or numpy.ndarray): Feature data
            y (pd.Series/DataFrame or numpy.ndarray): Target data
            feature_columns (list, optional): List of feature column names (for DataFrame input)
        """
        if isinstance(x, pd.DataFrame):
            # DataFrame input - extract features
            if feature_columns is None:
                # Use all columns as features
                feature_columns = x.columns.tolist()
            
            self.feature_columns = feature_columns
            
            # Extract feature data
            self.x_data = x[feature_columns].values.astype(np.float32)
                
        else:
            # Numpy array input (backward compatibility)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            self.x_data = x.astype(np.float32)
            
        # Handle target data
        if hasattr(y, 'values'):  # pandas Series/DataFrame
            self.y_data = y.values.astype(np.int64)
        else:  # numpy array
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            self.y_data = y.astype(np.int64)
        
        # Validate dimensions
        if len(self.x_data) != len(self.y_data):
            raise ValueError(f"X and y must have same number of samples. Got X: {len(self.x_data)}, y: {len(self.y_data)}")
 
    def __len__(self):
        return len(self.x_data)
 
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (features, target, index, identifier) as torch tensors
                   Note: identifier is the same as idx for confidence-aware learning
        """
        x_sample = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y_sample = torch.tensor(self.y_data[idx], dtype=torch.long)
        
        # Use idx as the identifier for confidence-aware learning
        identifier = idx
        
        return x_sample, y_sample, idx, identifier



def get_feature_list(feature_path, train_data_all, target):
    """
    Load feature list from file or create from dataframe with comprehensive error handling.
    
    Args:
        feature_path (str): Path to the feature list file
        train_data_all (pd.DataFrame): The training dataframe for fallback
        target (str or list): Target column name(s) to exclude from fallback
        
    Returns:
        list: Feature list loaded from file or created from dataframe columns
    """
    print("Attempting to load feature list...")
    
    # Check if feature file exists
    if not os.path.exists(feature_path):
        print(f"Feature file not found: {feature_path}")
        print("Creating feature list from dataframe columns")
        return create_feature_list_from_dataframe(train_data_all, target)
    
    # File exists, try to load it
    try:
        print(f"Feature file found: {feature_path}")
        feat_list = load_feature_list_from_file(feature_path)
        print(f"Successfully loaded feature list with {len(feat_list)} features")
        return feat_list
        
    except Exception as e:
        print(f"Failed to load feature list from file: {e}")
        print("Fallback: Creating feature list from dataframe columns")
        return create_feature_list_from_dataframe(train_data_all, target)


def load_feature_list_from_file(feature_path):
    """
    Load feature list from various file formats (pickle, parquet, or csv).
    Args:
        feature_path (str): Path to the feature list file
    Returns:
        list: List of feature names
    Raises:
        ValueError: If file format is not supported or content is invalid
    """
    _, ext = os.path.splitext(feature_path.lower())
    
    try:
        if ext in ['.pkl', '.pickle']:
            with open(feature_path, 'rb') as f:
                feat_data = pickle.load(f)
                
            # Handle different pickle content types
            if isinstance(feat_data, list):
                return feat_data
            elif isinstance(feat_data, dict):
                # Try common keys for feature lists
                for key in ['features', 'feature_list', 'columns', 'feat_list']:
                    if key in feat_data:
                        return feat_data[key]
                # If no standard key found, try first value that's a list
                for value in feat_data.values():
                    if isinstance(value, list):
                        return value
                raise ValueError("Dictionary doesn't contain recognizable feature list")
            elif hasattr(feat_data, 'columns'):  # DataFrame-like object
                return feat_data.columns.tolist()
            else:
                raise ValueError(f"Unsupported pickle content type: {type(feat_data)}")
                
        elif ext == '.parquet':
            df = pd.read_parquet(feature_path)
            return df.iloc[:, 0].tolist()  # First column contains feature names
            
        elif ext == '.csv':
            df = pd.read_csv(feature_path)
            return df.iloc[:, 0].tolist()  # First column contains feature names
            
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .pkl/.pickle, .parquet, .csv")
            
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        else:
            raise ValueError(f"Error reading {feature_path}: {str(e)}")


def create_feature_list_from_dataframe(train_data_all, target):
    """
    Create feature list from dataframe columns, excluding target column(s).
    Args:
        train_data_all (pd.DataFrame): The training dataframe
        target (str or list): Target column name(s) to exclude
    
    Returns:
        list: Feature list with target columns removed
    """
    feat_list = train_data_all.columns.tolist()
    
    # Remove target column(s) from feature list
    if isinstance(target, list):
        # Multi-label case: target is a list of column names
        feat_list = [col for col in feat_list if col not in target]
        print(f"Removed {len(target)} target columns for multi-label classification")
    else:
        # Single-label case: target is a single column name
        if target in feat_list:
            feat_list.remove(target)
            print(f"Removed target column '{target}' for single-label classification")
    
    print(f"Created feature list with {len(feat_list)} features from dataframe")
    return feat_list


def calculate_class_weights(y_train, target):
    """
    Calculate class weights for both multiclass and multilabel scenarios.
    Args:
        y_train: Training labels (pandas Series/DataFrame or numpy array)
        target (str or list): Target column name(s) to determine the scenario
    Returns:
        dict: Class weights for CrossEntropyLoss (multiclass) or pos_weight for BCEWithLogitsLoss (multilabel)
    """
    # Convert pandas Series/DataFrame to numpy for consistent handling if needed
    if hasattr(y_train, 'values'):
        y_values = y_train.values
    else:
        y_values = y_train
    
    if isinstance(target, list):
        # Multi-label classification: calculate normalized balanced weights for each label
        print("\n=== Multi-label Classification Weight Calculation ===")
        pos_weights = {}
        
        # Handle both DataFrame and numpy array cases
        if hasattr(y_train, 'shape') and len(y_train.shape) > 1:
            n_labels = y_train.shape[1]
            for i in range(n_labels):
                if hasattr(y_train, 'iloc'):
                    label_values = y_train.iloc[:, i].values  # Convert to numpy
                    label_name = target[i] if i < len(target) else f"label_{i}"
                else:
                    label_values = y_train[:, i]
                    label_name = target[i] if i < len(target) else f"label_{i}"
                
                neg_count = (label_values == 0).sum()
                pos_count = (label_values == 1).sum()
                total_samples = len(label_values)
                n_classes = 2  # Binary classification for each label
                
                # Calculate balanced class weights (same formula as multiclass)
                weight_neg = total_samples / (n_classes * neg_count) if neg_count > 0 else 0
                weight_pos = total_samples / (n_classes * pos_count) if pos_count > 0 else 0
                
                # pos_weight for BCEWithLogitsLoss is the ratio of positive to negative weight
                # This normalizes the positive class weight relative to the negative class baseline
                pos_weight = weight_pos / weight_neg if weight_neg > 0 else 1.0
                pos_weights[label_name] = float(pos_weight)
                
                print(f"{label_name}: pos_samples={pos_count}, neg_samples={neg_count}, weight_neg={weight_neg:.4f}, weight_pos={weight_pos:.4f}, pos_weight={pos_weight:.4f}")
        
        print(f"Multi-label pos_weights: {pos_weights}")
        return {'type': 'multilabel', 'pos_weights': pos_weights}
    
    else:
        # Single-label multiclass classification: calculate class weights
        print("\n=== Single-label Multiclass Weight Calculation ===")
        
        # Handle both Series and numpy array cases
        if hasattr(y_train, 'value_counts'):
            # Pandas Series - use value_counts for efficiency
            class_counts = y_train.value_counts().sort_index()
            total_samples = len(y_train)
        else:
            # Numpy array - use np.unique
            unique, counts = np.unique(y_values, return_counts=True)
            class_counts = dict(zip(unique, counts))
            total_samples = len(y_values)
        
        n_classes = len(class_counts)
        
        # Calculate inverse frequency weights
        class_weights = {}
        for class_idx, count in class_counts.items():
            weight = total_samples / (n_classes * count)
            class_weights[int(class_idx)] = float(weight)
            print(f"Class {class_idx}: samples={count}, weight={weight:.4f}")
        
        print(f"Class weights: {class_weights}")
        return {'type': 'multiclass', 'class_weights': class_weights}


def smart_read_data(file_path):
    """
    Intelligently read data files, handling parquet files with or without extensions.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    # Get the file extension (if any)
    _, ext = os.path.splitext(file_path.lower())
    
    # Check if it's explicitly a parquet file
    if ext == '.parquet':
        return pd.read_parquet(file_path)
    
    # If no extension or unknown extension, try to detect the format
    if ext == '' or ext not in ['.csv', '.parquet']:
        # Try reading as parquet first (common for files without extension)
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to read as parquet: {e}")
            print("Trying to read as CSV...")
            return pd.read_csv(file_path)
    
    # Default to CSV for .csv files or any other recognized text format
    return pd.read_csv(file_path)