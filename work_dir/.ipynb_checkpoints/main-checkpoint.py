import sys
import os
import math
import time

from nn_tab.logger import setup_logger
from nn_tab.datasets import read_train_data
from nn_tab.config_loader import load_config
from nn_tab.utils import train_model, train_model_crl
from nn_tab.models import get_model

import torch
torch.manual_seed(0)
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config=None):
    """Main training function that loads config and runs the complete training pipeline."""
    # Load config from file or use provided config
    if config is None:
        config = load_config()
    
    # Initialize logger
    exp_name = config.get("exp_name", "nn_on_tab_data")
    logger = setup_logger(name="nntab", exp_name=exp_name)
    logger.info("Starting neural network tabular training pipeline - will load data, initialize model, and train based on configuration")
    
    epochs = config.get("epochs", 100)
    train_path = config.get("train_path")
    test_path = config.get("test_path", "None")

    target = config.get("target", "fraud_sw")
    output_dim = config.get("output_dim", 2)
    batch_size = config.get("batch_size", 256)
    num_workers = config.get("num_workers", 0)
    rank_weight = config.get("rank_weight", 0.5)
    rank_weight_f = config.get("rank_weight_f", 0.5)

    hidden_layers = config.get("hidden_layers", [512, 128, 64, 4])
    feature_path = config.get("feature_path", "None")
    model_name = config.get("model_name", "fraudmodel_5layer")
    optimizer_config = config.get("optimizer", {"name": "Adam", "lr": 1e-4, "weight_decay": 1e-5})
    training_method = config.get("training_method", "standard_multiclass")
    loss_function = config.get("loss_function", None)
    focal_loss_config = config.get("focal_loss_config", {"gamma": 2.0, "alpha": None})
    use_class_weights = config.get("use_class_weights", True)
    
    # Load and preprocess data
    train_loader, val_loader, class_weights, feat_list = read_train_data(feature_path, target, train_path, batch_size, num_workers, training_method, use_class_weights)
    
    # Initialize model, loss, optimizer
    logger.info("Initializing model and training components")
    model, criterion, optimizer, scheduler = get_model(device, model_name, input_dim=len(feat_list), class_weights=class_weights, hidden_layers=hidden_layers, optimizer_config=optimizer_config, training_method=training_method, output=output_dim, loss_function=loss_function, focal_loss_config=focal_loss_config)

    start_time = time.time()
    logger.info("Training started!!")

    if training_method == "confidence_aware":
        logger.info(f"Using confidence-aware training with classification and ranking losses")
        train_model_crl(model, epochs, optimizer, scheduler, train_loader, val_loader, rank_weight, rank_weight_f, criterion, device, exp_name, model_name, training_method)
    
    elif training_method == "standard_multiclass":
        logger.info(f"Using standard training with single multiclass loss function")
        train_model(model, epochs, optimizer, scheduler, train_loader, val_loader, criterion, device, exp_name, model_name, training_method)
    
    elif training_method == "standard_multilabel":
        logger.info(f"Using standard training with multilabel loss function")
        train_model(model, epochs, optimizer, scheduler, train_loader, val_loader, criterion, device, exp_name, model_name, training_method)
    
    else:
        supported_methods = ["confidence_aware", "standard_multiclass", "standard_multilabel"]
        logger.error(f"Error: Training method '{training_method}' is not supported.")
        logger.error(f"Supported methods are: {supported_methods}")
        exit(1)

    # Record the end time
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    logger.info(f"Training completed successfully! Elapsed time in hours: {elapsed_time} hours")

if __name__ == "__main__":
    main()
  