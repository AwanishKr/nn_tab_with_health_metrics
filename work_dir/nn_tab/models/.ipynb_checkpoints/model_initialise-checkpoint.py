import torch
from torch import nn
import torch.nn.functional as F
from .model import fraudmodel_3layer, fraudmodel_5layer, fraudmodel_7layer, fraudmodel_8layer
from ..logger import get_logger


def focal_loss(inputs, targets, gamma=2.0, alpha=None, reduction='mean'):
    """Focal Loss: FL(pt) = -α(1-pt)^γ * log(pt). Handles class imbalance."""
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    p_t = F.softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_loss = (1 - p_t) ** gamma * ce_loss
    
    # Apply alpha balancing factor if provided
    if alpha is not None:
        if isinstance(alpha, (float, int)):
            focal_loss = alpha * focal_loss
        else:
            # Alpha is a tensor of per-class weights
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
    
    return focal_loss.mean() if reduction == 'mean' else focal_loss.sum() if reduction == 'sum' else focal_loss


class FocalLoss(nn.Module):
    """FocalLoss as nn.Module."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        return focal_loss(inputs, targets, self.gamma, self.alpha, self.reduction)




def _create_loss_function(loss_name, class_weights, device, output_dim, is_multilabel=False, focal_loss_config={"gamma": 2.0}):
    """Create loss function with optional class weighting. 
    Supports CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, FocalLoss.
    """
    logger = get_logger()
    
    if loss_name == "CrossEntropyLoss":
        if 'class_weights' in class_weights and class_weights['class_weights']:
            class_weights_list = [class_weights['class_weights'][i] for i in sorted(class_weights['class_weights'].keys())]
            weight_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor).to(device)
            logger.info(f"Using CrossEntropyLoss with class_weights: {class_weights_list}")
        else:
            criterion = nn.CrossEntropyLoss().to(device)
            logger.info("Using standard CrossEntropyLoss (no class_weights available)")
    
    elif loss_name == "BCEWithLogitsLoss":
        if is_multilabel and 'pos_weights' in class_weights and class_weights['pos_weights']:
            pos_weights_list = [class_weights['pos_weights'][label] for label in sorted(class_weights['pos_weights'].keys())]
            pos_weight_tensor = torch.tensor(pos_weights_list, dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
            logger.info(f"Using BCEWithLogitsLoss with pos_weights: {pos_weights_list}")
        else:
            criterion = nn.BCEWithLogitsLoss().to(device)
            logger.info("Using standard BCEWithLogitsLoss (no pos_weights available)")
    
    elif loss_name == "MSELoss":
        criterion = nn.MSELoss().to(device)
        logger.info("Using MSELoss")
    
    elif loss_name == "FocalLoss":
        gamma = focal_loss_config.get("gamma", 2.0)
        alpha = focal_loss_config.get("alpha", None)
        
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32).to(device) 
            criterion = FocalLoss(gamma=gamma, alpha=alpha).to(device)
            logger.info(f"Using FocalLoss (gamma={gamma}, alpha={alpha})")
        else:
            criterion = FocalLoss(gamma=gamma).to(device)
            logger.info(f"Using FocalLoss (gamma={gamma}, alpha=None)")
    
    else:
        logger.warning(f"Unknown loss function '{loss_name}', defaulting to CrossEntropyLoss")
        return _create_loss_function("CrossEntropyLoss", class_weights, device, output_dim, is_multilabel, focal_loss_config)
    
    return criterion




def get_model(device, model_name, input_dim, class_weights, hidden_layers=[512, 128, 64, 4], optimizer_config={"name": "Adam", "lr": 1e-4, "weight_decay": 1e-5}, training_method="standard_multiclass", output=2, loss_function=None, focal_loss_config={"gamma": 2.0}):
    """
    Initialize model, loss, optimizer, and scheduler.
    
    Returns: (model, criterion, optimizer, scheduler)
    For confidence_aware: criterion is tuple (base_loss, ranking_loss)
    """
    logger = get_logger()
    logger.info("Initializing model architecture, loss functions, optimizer, and scheduler based on training method and configuration")
    
    # Model selection based on model_name parameter
    model_mapping = {
                        'fraudmodel_3layer': fraudmodel_3layer,
                        'fraudmodel_5layer': fraudmodel_5layer,
                        'fraudmodel_7layer': fraudmodel_7layer,
                        'fraudmodel_8layer': fraudmodel_8layer
                    }
    
    if model_name not in model_mapping:
        logger.warning(f"Model '{model_name}' not found. Defaulting to 'fraudmodel_5layer'")
        model_name = 'fraudmodel_5layer'
    
    # Initialize the selected model with hidden_layers list
    model_class = model_mapping[model_name]
    model = model_class(input_dim, hidden_layers=hidden_layers, output=output)
    
    logger.info(f"Using model: {model_name} with hidden layers: {hidden_layers}")
    logger.info(f"Training method: {training_method}")
    
    # ============================================================ LOSS FUNCTION SETUP ============================================================
    # Step 1: Determine the base loss function
    if loss_function:
        base_loss_name = loss_function
        logger.info(f"Using user-specified base loss function: {base_loss_name}")
    else:
        # Determine default base loss based on training method
        if training_method == "multilabel":
            base_loss_name = "BCEWithLogitsLoss"
        else:  # standard_multiclass, confidence_aware, or default
            base_loss_name = "CrossEntropyLoss"
        logger.info(f"Using default base loss function for {training_method}: {base_loss_name}")
    
    # Step 2: Create the base loss with appropriate settings
    is_multilabel = (training_method == "multilabel")
    base_criterion = _create_loss_function(base_loss_name, class_weights, device, output, is_multilabel=is_multilabel, focal_loss_config=focal_loss_config)
    
    # Step 3: Apply training method-specific logic
    if training_method == "confidence_aware":
        # Confidence-aware: base loss + ranking loss
        criterion_ranking = nn.MarginRankingLoss(margin=0.0).to(device)
        criterion = (base_criterion, criterion_ranking)  # Return tuple
        logger.info(f"Confidence-aware training: {base_loss_name} (classification) + MarginRankingLoss (ranking)")
    else:
        # Standard multiclass or multilabel: use base loss only
        criterion = base_criterion
        logger.info(f"Using single loss function: {base_loss_name}")
    # ============================================================ END LOSS FUNCTION SETUP ============================================================
    
    # Dynamic optimizer selection
    optimizer_name = optimizer_config.get("name", "Adam")
    learning_rate = optimizer_config.get("lr", 1e-4)
    weight_decay = optimizer_config.get("weight_decay", 1e-5)
    
    optimizer_mapping = {
                            "Adam": torch.optim.Adam,
                            "SGD": torch.optim.SGD,
                            "RMSprop": torch.optim.RMSprop,
                            "AdamW": torch.optim.AdamW,
                            "Adamax": torch.optim.Adamax
                        }
    
    if optimizer_name not in optimizer_mapping:
        logger.warning(f"Optimizer '{optimizer_name}' not found. Defaulting to 'Adam'")
        optimizer_name = "Adam"
    
    # Create optimizer with different parameters based on type
    if optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        optimizer = optimizer_mapping[optimizer_name](model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optimizer_mapping[optimizer_name](model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
    logger.info(f"Using optimizer: {optimizer_name} with lr={learning_rate}, weight_decay={weight_decay}")
    logger.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    return model, criterion, optimizer, scheduler