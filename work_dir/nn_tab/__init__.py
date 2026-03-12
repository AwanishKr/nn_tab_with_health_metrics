"""NNTab Library - Neural Network Tabular Data Processing

A comprehensive library for training neural networks on tabular data with:
- data loading and preprocessing
- Multiple neural network architectures
- Balanced training with class weights
- Comprehensive evaluation and visualization
"""

# import builtins
# builtins.print = lambda *args, **kwargs: __import__('builtins').print(*args, **{**kwargs, 'flush': True})

# Import from submodules
from .datasets import read_train_data, RobustScaleSmoothClipTransform, smart_read_data
from .models import get_model, fraudmodel_3layer, fraudmodel_5layer, fraudmodel_7layer, fraudmodel_8layer
from .utils import train_model, train_model_crl, val_fn, train_fn, check_for_invalid_values
from .plots import auc_plot, plot_loss_curve, tnxs_plots
from .config_loader import load_config
from .logger import setup_logger, get_logger, LoggerMixin

# Import submodules for module-level access
from . import datasets
from . import models
from . import utils
from . import plots
from . import config_loader

__version__ = "0.1.0"
__author__ = "Awanish Kumar"

# Main public API - what users see with tab completion
__all__ = [
    # Core workflow functions
    'read_train_data',      # Data loading and preprocessing
    'get_model',            # Model, optimizer, criterion initialization
    'train_model',          # Complete training pipeline (standard)
    'train_model_crl',      # Complete training pipeline (curriculum learning)
    'load_config',          # Configuration loading contains parameters for training
    
    # Logging utilities
    'setup_logger',         # Logger setup function
    'get_logger',           # Get existing logger
    'LoggerMixin',          # Mixin class for logging
    
    # Individual training functions
    'val_fn',               # Validation function
    'train_fn',             # Training function
    
    # Model architectures
    'fraudmodel_3layer',    # 3-layer neural network
    'fraudmodel_5layer',    # 5-layer neural network (recommended)
    'fraudmodel_7layer',    # 7-layer neural network
    'fraudmodel_8layer',    # 8-layer neural network
    
    # Data preprocessing
    'RobustScaleSmoothClipTransform',  # Data transformation
    'smart_read_data',      # Smart data loading
    
    # Evaluation and visualization
    'plot_loss_curve',      # Training loss visualization
    'auc_plot',            # AUC ROC plotting
    'tnxs_plots',          # Business KPI plots
    
    # Utilities
    'check_for_invalid_values',  # Data validation
    
    # Submodules
    'datasets',
    'models', 
    'utils',
    'plots',
    'config_loader'
]

def help():
    """Display comprehensive help information about the nntab package."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NNTab Library - Neural Network Tabular Data Processing    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 AVAILABLE MODULES:

┌─ 📊 nntab.datasets - Data Loading & Preprocessing
│  ├─ read_train_data()              - Complete data pipeline: load, preprocess, split
│  ├─ smart_read_data()              - Auto-detect and load CSV/Parquet files
│  ├─ RobustScaleSmoothClipTransform - Advanced data normalization transformer
│  ├─ get_feature_list()             - Load/generate feature lists from files
│  └─ clean_date()                   - Clean and decode date string formats
│
┌─ 🧠 nntab.models - Neural Network Architectures & Initialization  
│  ├─ get_model()                    - Initialize model + optimizer + loss (with class weights)
│  ├─ fraudmodel_3layer()            - Lightweight 3-layer architecture
│  ├─ fraudmodel_5layer()            - ⭐ Recommended 5-layer architecture
│  ├─ fraudmodel_7layer()            - Deep 7-layer architecture
│  └─ fraudmodel_8layer()            - Very deep 8-layer architecture
│
┌─ 🔧 nntab.utils - Training & Validation Utilities
│  ├─ train_model()                  - Complete training pipeline with early stopping
│  ├─ train_fn()                     - Single epoch training function
│  ├─ val_fn()                       - Validation function with metrics
│  └─ check_for_invalid_values()     - Data validation (NaN/Inf detection)
│
┌─ 📈 nntab.plots - Visualization & Evaluation
│  ├─ plot_loss_curve()              - Training/validation loss visualization
│  ├─ auc_plot()                     - ROC-AUC comparison plots
│  ├─ tnxs_plots()                   - Business KPI analysis plots
│  ├─ plot_info()                    - Calculate business metrics across thresholds
│  ├─ ttnr_tdr()                     - True TNR vs TDR plots
│  ├─ ttnr_tfpr()                    - True TNR vs TFPR plots
│  └─ ttnr_fraud_bps()               - TNR vs Fraud Basis Points plots
│
└─ ⚙️  nntab.config_loader - Configuration Management
   └─ load_config()                  - Load training parameters from config.json file

🚀 QUICK START WORKFLOW:

  import nntab
  
  # 1. Load configuration (reads config.json automatically)
  config = nntab.load_config()
  
  # 2. Load and preprocess data
  train_loader, val_loader, class_weights, feat_list = nntab.read_train_data(
      feature_path=config['feature_path'],
      target=config['target'], 
      file_path=config['train_path'],
      batch_size=config['batch_size'],
      num_workers=config['num_workers']
  )
  
  # 3. Initialize model with automatic class weighting
  model, criterion, optimizer, scheduler = nntab.get_model(
      device='cuda',
      input_dim=len(feat_list),
      class_weights=class_weights
  )
  
  # 4. Train with validation and early stopping
  nntab.train_model(model, epochs=100, optimizer, scheduler, 
                   train_loader, val_loader, criterion, device, exp_name)

💡 IMPORT STYLES:

  # Direct access (recommended for main functions)
  from nntab import read_train_data, get_model, train_model
  
  # Module-specific imports
  from nntab.datasets import smart_read_data
  from nntab.models import fraudmodel_5layer
  from nntab.plots import auc_plot
  from nntab.utils import val_fn
  
  # Module access
  nntab.datasets.read_train_data(...)
  nntab.models.fraudmodel_5layer(...)

📚 For detailed function documentation: help(nntab.function_name)
🔍 To explore: dir(nntab) or dir(nntab.module_name)
    """)

# Make help easily discoverable
__all__.append('help')