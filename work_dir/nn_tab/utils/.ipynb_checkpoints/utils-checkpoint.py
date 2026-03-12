import os
import sys
import torch
import torch.nn.functional as F
import logging
import pandas as pd
import pickle as pkl

from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from .calculate_scores import EL2N_score, calculate_aum, update_forgetting, add_logits_to_aum_dict
from ..plots import plot_loss_curve
from .crl_utils import ForgettingTracker, History
from ..logger import get_logger


def accuracy_metric(output, target):
    """Calculate accuracy for classification task.
    
    Args:
        output: Model predictions (logits)
        target: Ground truth labels
    
    Returns:
        tuple: (accuracy_percentage, correct_predictions_tensor)
    """
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).float()
        accuracy_val = correct.mean().item() * 100.0
        return accuracy_val, correct


class AverageMeter(object):
    """Computes and stores the average and current value for tracking metrics."""
    
    def __init__(self, track_history=False):
        """Initialize the meter.
        
        Args:
            track_history: If True, maintains a history of values for plotting
        """
        self.track_history = track_history
        self.reset()

    def reset(self):
        """Reset all values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.track_history:
            self.history = []

    def update(self, val, n=1):
        """Update the meter with a new value.
        
        Args:
            val: New value to add
            n: Number of samples (default 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        if self.track_history:
            self.history.append(val)
    
    def get_history(self):
        """Get the history of values for plotting.
        
        Returns:
            list: History of values if track_history=True, empty list otherwise
        """
        return self.history if self.track_history else []


def check_for_invalid_values(X, y_hat, loss, context=""):
    """
    Check for NaN and inf values in tensors and return True if any invalid values found.
    
    Args:
        X: Input tensor
        y_hat: Prediction tensor
        loss: Loss tensor
        context: String to identify where the check is called from (e.g., "training", "validation")
    
    Returns:
        bool: True if invalid values found, False otherwise
    """
    logger = get_logger()
    
    if torch.isnan(X).any() or torch.isinf(X).any():
        logger.error(f"X_batch in {context} contains NaN or inf values!")
        return True
    
    if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
        logger.error(f"y_hat in {context} contains NaN or inf values!")
        return True
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        logger.error(f"Loss on {context} set contains NaN or inf values!")
        return True
    
    return False


def val_fn(model, data_loader, device, criterion, training_method):
    """Validation function that evaluates model performance on validation set.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for validation data
        device: Device to run evaluation on (cuda/cpu)
        criterion: Loss function to use
        
    Returns:
        tuple: (average_loss, accuracy, auc_pr_score)
    """
    
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    y_true = []
    y_pred_probs = []
    
    with torch.no_grad():
        if training_method == 'confidence_aware':
            for X, y,_,_ in tqdm(data_loader, desc="Validation", leave=False, disable=True):
                X = X.to(device)
                y = y.to(device)
                
                y_hat = model(X)
                loss = criterion(y_hat, y)
                
                # Check for NaN or Inf values in inputs, outputs, and loss
                if check_for_invalid_values(X, y_hat, loss, "validation"):
                    break
                
                # Accuracy computation: get predicted class from y_hat
                _, predicted = torch.max(y_hat, 1)
                batch_accuracy = (predicted == y).sum().item() / y.size(0)
                
                # Update meters with proper weighting
                losses.update(loss.item(), y.size(0))
                accuracies.update(batch_accuracy, y.size(0))
                
                # Collect ground truth and predicted probabilities (for class 1)
                y_true.extend(y.cpu().numpy())
                # Use softmax to compute probabilities, then take class 1 probability.
                y_pred_probs.extend(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
        else:
            for X, y in tqdm(data_loader, desc="Validation", leave=False, disable=True):
                X = X.to(device)
                y = y.to(device)
                
                y_hat = model(X)
                loss = criterion(y_hat, y)
                
                # Check for NaN or Inf values in inputs, outputs, and loss
                if check_for_invalid_values(X, y_hat, loss, "validation"):
                    break
                
                # Accuracy computation: get predicted class from y_hat
                _, predicted = torch.max(y_hat, 1)
                batch_accuracy = (predicted == y).sum().item() / y.size(0)
                
                # Update meters with proper weighting
                losses.update(loss.item(), y.size(0))
                accuracies.update(batch_accuracy, y.size(0))
                
                # Collect ground truth and predicted probabilities (for class 1)
                y_true.extend(y.cpu().numpy())
                # Use softmax to compute probabilities, then take class 1 probability.
                y_pred_probs.extend(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
    
    aucpr = average_precision_score(y_true, y_pred_probs)
    return losses.avg, accuracies.avg, aucpr


def train_fn(model, data_loader, optimizer, device, criterion):
    """Training function for a single epoch.
    
    Args:
        model: PyTorch model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        device: Device to run training on (cuda/cpu)
        criterion: Loss function to use
        
    Returns:
        tuple: (average_loss, accuracy)
    """

    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for X, y in tqdm(data_loader, desc="Training", leave=False, disable=True):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y)
        
        # Check for NaN or inf values
        if check_for_invalid_values(X, y_hat, loss, "training"):
            break
        
        # Calculate batch accuracy assuming a classification task
        _, predicted = torch.max(y_hat, 1)
        batch_accuracy = (predicted == y).sum().item() / y.size(0)
        
        # Update meters with proper weighting
        losses.update(loss.item(), y.size(0))
        accuracies.update(batch_accuracy, y.size(0))
        
        loss.backward()
        optimizer.step()
    
    return losses.avg, accuracies.avg



def train_crl(loader, model, criterion_cls, criterion_ranking, optimizer, epoch, chistory, fhistory, rank_weight, rank_weight_f, aum_dict, el2n_scores, forgetting_scores, device):
    """Curriculum Learning training function for confidence-aware training.
    
    Args:
        loader: DataLoader for training data
        model: PyTorch model to train
        criterion_cls: Classification loss function
        criterion_ranking: Ranking loss function for confidence learning
        optimizer: Optimizer for model parameters
        epoch: Current epoch number
        chistory: History object for correctness tracking
        fhistory: ForgettingTracker object for forgetting events
        rank_weight: Weight for ranking loss
        rank_weight_f: Weight for forgetting-based ranking
        aum_dict: Dictionary for AUM scores
        el2n_scores: Dictionary for EL2N scores
        forgetting_scores: Dictionary for forgetting scores
        device: Device to run training on (cuda/cpu)
        
    Returns:
        tuple: Updated score dictionaries and metrics
    """
    logger = get_logger()
    
    # Track progress and performance
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # top1 = AverageMeter()
    cls_losses = AverageMeter()
    ranking_losses = AverageMeter()
    
    model.to(device)
    model.train()

    total_train_loss = 0
    total_correct = 0
    total_samples = 0
    all_train_preds = []
    all_train_targets = []

    for inputx, target, idx, identifier in loader:
        inputx, target = inputx.to(device, dtype=torch.float32), target.to(device)
        output = model(inputx)

        # Compute ranking target value normalization
        conf = F.softmax(output, dim=1)
        confidence, _ = conf.max(dim=1)

        rank_input1 = confidence  #[012345]
        rank_input2 = torch.roll(confidence, -1) #[123450]
        idx2 = torch.roll(idx, -1)

        # Get rank target and margin
        rank_target, rank_margin = chistory.get_target_margin(idx, idx2)     # target if ci>cj or vv and margin ci-cj
        rank_targetc, rank_marginc = rank_target.to(device), rank_margin.to(device)
        
        # print(idx,idx2)
        # Get rank target and margin for forgetting
        rank_target, rank_margin = fhistory.get_forgetting_target_margin(idx, idx2) 
        rank_targetf, rank_marginf = rank_target.to(device), rank_margin.to(device)
        
        # Dump dictionary for specific epochs only
        # if epoch in [10, 50, 90]:
            # dump_dictionary(epoch, rank_input1, rank_input2, rank_target,idx, idx2, target, _)

        rank_target_nonzeroc = rank_targetc.clone()
        rank_target_nonzeroc[rank_target_nonzeroc == 0] = 1
        rank_inputc2 = rank_input2 + rank_marginc / rank_target_nonzeroc
        
        rank_target_nonzerof = rank_targetf.clone()
        rank_target_nonzerof[rank_target_nonzerof == 0] = 1
        rank_inputf2 = rank_input2 + rank_marginf / rank_target_nonzerof

        # Compute ranking loss
        ranking_loss = criterion_ranking(rank_input1, rank_inputc2, rank_targetc)
        ranking_loss_f = criterion_ranking(rank_input1, rank_inputf2, rank_targetf)

        # Compute classification loss and total loss
        cls_loss = criterion_cls(output, target)
        
        # final loss
        loss = cls_loss + rank_weight * ranking_loss + rank_weight_f * ranking_loss_f

        # calculate scores
        aum_dict = calculate_aum(output.detach().cpu(), target.detach().cpu(), identifier.tolist(), aum_dict, epoch)
        el2n_scores = EL2N_score(conf.detach().cpu(), target.detach().cpu(), identifier.tolist(), el2n_scores, epoch)
        forgetting_scores = update_forgetting(output.detach().cpu(), target.detach().cpu(), identifier.tolist(), epoch, forgetting_scores)

         # Check for NaN or inf values
        if check_for_invalid_values(inputx, output, loss, "training"):
            break

        # Calculate batch accuracy assuming a classification task
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        all_train_preds.append(confidence.detach())
        all_train_targets.append(target.detach())

        # Track performance metrics
        prec, correct = accuracy_metric(output, target)
        cls_losses.update(cls_loss.item(), inputx.size(0))
        ranking_losses.update(ranking_loss.item(), inputx.size(0))
        # total_train_loss.update(loss.item(), inputx.size(0))
        # top1.update(prec.item(), inputx.size(0))

        # Update correctness history
        chistory.correctness_update(idx, correct, output)
        fhistory.forgetting_update(output.detach().cpu(), target.cpu(), idx, epoch)
        
    logger.info(f'Epoch [{epoch}] CRL Training Summary - '
                f'Cls Loss: {cls_losses.avg:.4f} | '
                f'Rank Loss: {ranking_losses.avg:.4f} | '
                f'Total Batches: {len(loader)}')
    
    # Update max correctness for epoch
    chistory.max_correctness_update(epoch)
    fhistory.max_forgetting_update(epoch)
    
    all_train_preds = torch.cat([t.cpu() for t in all_train_preds]).numpy()
    all_train_targets = torch.cat([t.cpu() for t in all_train_targets]).numpy()
    avg_loss = total_train_loss / len(loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    try:
        auroc = roc_auc_score(all_train_targets, all_train_preds)
    except ValueError:
        auroc = float('nan')

    return avg_loss, accuracy, aum_dict, el2n_scores, forgetting_scores


def train_model_crl(model, epochs, optimizer, scheduler, train_loader, val_loader, rank_weight, rank_weight_f, criterion, device, exp_name, model_name, training_method):
    """Complete training pipeline for Curriculum Learning with confidence-aware loss.
    
    Args:
        model: PyTorch model to train
        epochs: Number of training epochs
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        rank_weight: Weight for ranking loss component
        rank_weight_f: Weight for forgetting-based ranking
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Tuple of (classification_loss, ranking_loss)
        device: Device to run training on (cuda/cpu)
        exp_name: Experiment name for saving outputs
        model_name: Model name for nested folder structure
    """
    logger = get_logger()
    model.to(device)
    logger.info(f"Training setup - Device: {device}, Experiment name: {exp_name}")
    
    # Use AverageMeter with history tracking for plotting
    train_losses = AverageMeter(track_history=True)
    val_losses = AverageMeter(track_history=True)
    lrs = []
    
    aum_dict = {}
    el2n_scores = {}
    forgetting_scores = {}
    # depth_scores = {}
    
    best_validation_loss = float('inf')
    max_epochs_without_improvement = 10
    counter = 0
    criterion_cls, criterion_ranking = criterion
            
    correctness_history = History(len(train_loader.dataset))
    forgetting_history = ForgettingTracker(num_examples=len(train_loader.dataset))
    
    # Create directory structure once before training loop
    history_metrics_dir = os.path.join(os.getcwd(), "history_metrics", exp_name)
    aum_dir = os.path.join(history_metrics_dir, "aum_scores")
    el2n_dir = os.path.join(history_metrics_dir, "el2n_scores")
    forgetting_dir = os.path.join(history_metrics_dir, "forgetting_scores")
    
    os.makedirs(aum_dir, exist_ok=True)
    os.makedirs(el2n_dir, exist_ok=True)
    os.makedirs(forgetting_dir, exist_ok=True)
    
    for epoch_num in range(1, epochs + 1):
        input_list = [train_loader, model, criterion_cls, criterion_ranking, optimizer, epoch_num, 
                      correctness_history, forgetting_history, rank_weight, rank_weight_f, aum_dict, el2n_scores, forgetting_scores, device
                      ]
        
        train_loss, train_accuracy, aum_dict, el2n_scores, forgetting_scores = train_crl(*input_list)
        val_loss, test_accuracy, test_aucpr = val_fn(model, val_loader, device, criterion_cls, training_method)

        if epoch_num % 1 == 0:
            # Save AUM scores
            AUM_PICKLE_PATH = os.path.join(aum_dir, f"aum_dict_{epoch_num}.pkl")
            with open(AUM_PICKLE_PATH, "wb") as f:
                pkl.dump(add_logits_to_aum_dict(model, train_loader, device, aum_dict), f)
                
            # Save EL2N scores
            el2n_PICKLE_PATH = os.path.join(el2n_dir, f"el2n_score_dict_{epoch_num}.pkl")
            with open(el2n_PICKLE_PATH, "wb") as f:
                pkl.dump(el2n_scores, f)
                
            # Save forgetting scores
            forgetting_scores_PICKLE_PATH = os.path.join(forgetting_dir, f"forgetting_scores_dict_{epoch_num}.pkl")
            with open(forgetting_scores_PICKLE_PATH, "wb") as f:
                pkl.dump(forgetting_scores, f)
    
            # GS_PICKLE_PATH = "grand_score_dict_"+str(epoch_num)+".pkl"
            # with open(GS_PICKLE_PATH, "wb") as f:
                # pkl.dump(grand_scores, f)

            # DEPTH_PICKLE_PATH = "depth_dict_folder/depth_dict"+str(epoch_num)+".pkl"
            # os.makedirs(os.path.dirname(DEPTH_PICKLE_PATH), exist_ok=True)

            # with open(DEPTH_PICKLE_PATH, "wb") as f:
            #     pkl.dump(depth_scores, f)

        scheduler.step(val_loss)
        lrs.append(optimizer.param_groups[0]["lr"])
        
        train_losses.update(train_loss)
        val_losses.update(val_loss)
        
        logger.info(f"New learning rate: {optimizer.param_groups[0]['lr']}")
        logger.info(f"Epoch={epoch_num}, Train Loss={train_loss}, Test Loss={val_loss} , test aucpr={test_aucpr}, Test Accuracy={test_accuracy}")
        
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            counter = 0
            
            model_dir = os.path.join(os.getcwd(), "models", exp_name, model_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_name + ".pth")            
            
            torch.save({'model_state_dict':model.state_dict()}, model_path)
            logger.info(f"Best model saved at: {model_path}")
        else:
            counter = counter + 1
            if counter == max_epochs_without_improvement:
                logger.info("Early stopping triggered - counter reached max_epochs_without_improvement, stopping training")
                break
            else:
                logger.info("Counter increased but haven't reached max_epochs_without_improvement, continuing training")
    plot_loss_curve(exp_name, model_name, train_losses.get_history(), val_losses.get_history())



def train_model(model, epochs, optimizer, scheduler, train_loader, val_loader, criterion, device, exp_name, model_name, training_method):
    """Complete training pipeline with validation and early stopping.
    
    Args:
        model: PyTorch model to train
        epochs: Number of training epochs
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function to use
        device: Device to run training on (cuda/cpu)
        exp_name: Experiment name for saving outputs
        model_name: Model name for nested folder structure
    """
    logger = get_logger()
    model.to(device)
    logger.info(f"Training setup - Device: {device}, Experiment name: {exp_name}")
    
    # Use AverageMeter with history tracking for plotting
    train_losses = AverageMeter(track_history=True)
    val_losses = AverageMeter(track_history=True)
    lrs = []
    
    best_validation_loss = float('inf')
    epochs_since_improvement = 0
    max_epochs_without_improvement = 10
    counter = 0
            
    for epoch in range(epochs):
        train_loss, train_accuracy = train_fn(model, train_loader, optimizer, device, criterion)
        val_loss, test_accuracy, test_aucpr = val_fn(model, val_loader, device, criterion, training_method)

        scheduler.step(val_loss)
        lrs.append(optimizer.param_groups[0]["lr"])
        
        train_losses.update(train_loss)
        val_losses.update(val_loss)
        
        logger.info(f"New learning rate: {optimizer.param_groups[0]['lr']}")
        logger.info(f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={val_loss} , test aucpr={test_aucpr}, Test Accuracy={test_accuracy}")
        
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            counter = 0
            
            ## Saving the best model
            model_dir = os.path.join(os.getcwd(), "models", exp_name, model_name)
            # Create dir if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)

            # Path to save model
            model_path = os.path.join(model_dir, model_name + ".pth")
            torch.save({'model_state_dict':model.state_dict()}, model_path)
            logger.info(f"Best model saved at: {model_path}")
            
        else:
            counter = counter + 1
            if counter == max_epochs_without_improvement:
                logger.info("Early stopping triggered - counter reached max_epochs_without_improvement, stopping training")
                break
            
            else:
                logger.info("Counter increased but haven't reached max_epochs_without_improvement, continuing training")    
    
    plot_loss_curve(exp_name, model_name, train_losses.get_history(), val_losses.get_history())
