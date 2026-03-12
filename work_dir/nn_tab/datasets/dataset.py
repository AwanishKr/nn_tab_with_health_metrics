import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler
import sklearn
import pickle
import os
from .preprocess import *
from .dataloading_helpers import *
from ..logger import get_logger


def read_train_data(feature_path, target, file_path, batch_size, num_workers, training_method='auto', use_class_weights=True):
    """Load and preprocess tabular data for neural network training.
    
    Args:
        feature_path: Path to feature list file or auto-generate from data
        target: Target column name(s)
        file_path: Path to training data (parquet/csv)
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        training_method: Training method ('standard_multiclass', 'multilabel', 'confidence_aware')
        use_class_weights: Whether to calculate class weights (default: True)
    
    Returns:
        tuple: (train_loader, val_loader, class_weights, feat_list)
    
    Note:
        For confidence_aware training, the dataset index (idx) is used as the identifier for tracking samples.
    """
    logger = get_logger()
    logger.info("Loading and preprocessing tabular data - will read file, clean data, split train/val, calculate class weights, normalize, and create DataLoaders")
    
    train_data_all = smart_read_data(file_path)
    logger.info(f"Received train file_path: {file_path}")
    logger.info(f"Train data shape: {train_data_all.shape}")

    # select only numerical features, create feature list and then replace nans and inf with 0
    # train_data_all = train_data_all.select_dtypes(exclude=['object'])
    feat_list = get_feature_list(feature_path, train_data_all, target)
    train_data_all = train_data_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Extract features and target - idx will be used as identifier for confidence_aware training
    X_train_data = train_data_all[feat_list]
    y_train_data = train_data_all[target]

    X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, test_size=0.10, random_state=44)
    logger.info(f"Train/validation split completed - Train: X{X_train.shape}, y{y_train.shape} | Validation: X{X_val.shape}, y{y_val.shape}")
    
    # Calculate class weights based on target type (if enabled)
    class_weights = calculate_class_weights(y_train, target) if use_class_weights else {}
    
    # Check NaN count in feature columns
    total_nan_count = X_train.isna().sum().sum()
    logger.info(f"Total NaN count in features: {total_nan_count}")
    logger.info("Starting the normalization process")
    
    # ---------------------------- apply the robust scaling smooth clipping transformation here ----------------------------
    # Use the enhanced transformer with DataFrame support
    transformer = RobustScaleSmoothClipTransform()
    transformer.fit(X_train, y_train, feature_columns=feat_list)
    
    # Transform preserves DataFrame structure, only scales feature columns
    X_train_transformed = transformer.transform(X_train, feature_columns=feat_list)
    X_val_transformed = transformer.transform(X_val, feature_columns=feat_list)
    logger.info("Normalization completed successfully")

    # Create datasets based on training method
    if training_method == 'confidence_aware':
        # Use Custom_Dataset_CRL which uses idx as identifier
        train_ds = Custom_Dataset_CRL(X_train_transformed, y_train, 
                                     feature_columns=feat_list)
        val_ds = Custom_Dataset_CRL(X_val_transformed, y_val,
                                   feature_columns=feat_list)
    else:
        # Standard approach: use Custom_Dataset_normal
        train_ds = Custom_Dataset_normal(X_train_transformed, y_train, 
                                        feature_columns=feat_list)
        val_ds = Custom_Dataset_normal(X_val_transformed, y_val, 
                                      feature_columns=feat_list)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, class_weights, feat_list