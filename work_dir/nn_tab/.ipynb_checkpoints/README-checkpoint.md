# Neural Network Tabular Data Package

A comprehensive Python package for training neural networks on tabular data with support for multiple training methods including **confidence-aware learning** with curriculum learning (CRL), standard multiclass/multilabel classification, and extensive experiment tracking.

## 🚀 Key Features

- 🎯 **Multiple Training Methods**: Standard multiclass, multilabel, and confidence-aware curriculum learning
- 📊 **Advanced Loss Functions**: CrossEntropyLoss, FocalLoss, BCEWithLogitsLoss, MSELoss
- 📈 **Health Metrics Tracking**: AUM, EL2N, and Forgetting scores for sample-level analysis
- 🔄 **Automatic Preprocessing**: Feature scaling, normalization, and class weight balancing
- 📉 **Early Stopping & Scheduling**: Patience-based stopping with ReduceLROnPlateau scheduler
- 🗂️ **Experiment Management**: Organized output structure per experiment
- 📝 **Comprehensive Logging**: All training events, metrics, and errors logged to file

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd nn_package_tabular-main

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib tqdm

# Optional: Install in development mode
pip install -e .
```

## 🏗️ Project Structure

```
nn_package_tabular-main/
├── config.json              # Training configuration
├── main.py                  # Entry point with exception handling
├── nn_tab/                  # Main package
│   ├── __init__.py
│   ├── config_loader.py     # Config file parsing
│   ├── logger.py            # Logging setup
│   ├── datasets/            # Data loading and preprocessing
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── dataloading_helpers.py
│   ├── models/              # Neural network architectures
│   │   ├── model.py
│   │   └── model_initialise.py
│   ├── plots/               # Visualization utilities
│   │   └── plots.py
│   └── utils/               # Training and evaluation
│       ├── utils.py
│       ├── calculate_scores.py
│       └── crl_utils.py
│
├── logs/                    # Training logs (auto-created)
│   └── {exp_name}_{timestamp}.log
├── Models/                  # Saved models (auto-created)
│   └── {exp_name}/
│       └── {model_name}/
│           └── {model_name}.pth
├── history_metrics/         # Training metrics (auto-created)
│   └── {exp_name}/
│       ├── aum_scores/
│       ├── el2n_scores/
│       └── forgetting_scores/
└── plots/                   # Training plots (auto-created)
    └── {exp_name}/
        └── {model_name}/
```

## ⚙️ Configuration

Edit `config.json` to customize your training:

```json
{
    "train_path": "path/to/train.parquet",
    "test_path": "path/to/test.parquet",
All training parameters are specified in `config.json`:

```json
{
    "train_path": "path/to/train.parquet",
    "test_path": "path/to/test.parquet",
    "feature_path": "path/to/features.pkl",
    "batch_size": 512,
    "output_dim": 2,
    "target": "fraud_sw",
    "hidden_layers": [512, 128, 64, 4],
    "epochs": 100,
    "model_name": "fraudmodel_5layer",
    "optimizer": {"name": "Adam", "lr": 1e-4, "weight_decay": 1e-5},
    "rank_weight": 0.5,
    "rank_weight_f": 0.5,
    "training_method": "confidence_aware",
    "loss_function": "CrossEntropyLoss",
    "focal_loss_config": {"gamma": 2.0, "alpha": null},
    "use_class_weights": true,
    "exp_name": "my_experiment"
}
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_path` | String | Path to training data file (parquet/csv) |
| `test_path` | String | Path to test data file (optional) |
| `feature_path` | String | Path to feature list file (.pkl/.parquet/.csv) |
| `batch_size` | Integer | Batch size for training/validation (default: 512) |
| `output_dim` | Integer | Number of output classes (e.g., 2 for binary) |
| `target` | String | Target column name in dataset |
| `hidden_layers` | List[Int] | Layer dimensions for neural network (e.g., [512, 128, 64, 4]) |
| `epochs` | Integer | Maximum number of training epochs |
| `model_name` | String | Model architecture name (`fraudmodel_3layer`, `fraudmodel_5layer`, `fraudmodel_7layer`, `fraudmodel_8layer`) |
| `optimizer` | Dict | Optimizer config: `{"name": "Adam/SGD/AdamW/RMSprop", "lr": float, "weight_decay": float}` |
| `rank_weight` | Float | Weight for correctness-based ranking loss (confidence_aware only) |
| `rank_weight_f` | Float | Weight for forgetting-based ranking loss (confidence_aware only) |
| `training_method` | String | Training method: `confidence_aware`, `standard_multiclass`, `standard_multilabel` |
| `loss_function` | String | Loss function: `CrossEntropyLoss`, `FocalLoss`, `BCEWithLogitsLoss`, `MSELoss` |
| `focal_loss_config` | Dict | Focal loss parameters: `{"gamma": float, "alpha": float or list}` |
| `use_class_weights` | Boolean | Enable automatic class weight balancing |
| `exp_name` | String | Experiment name for organizing all outputs |
| `num_workers` | Integer | Number of data loader workers (optional, default: 0)
python main.py
```

The script will:
1. Load configuration from `config.json`
2. Initialize logging to `logs/{exp_name}_{timestamp}.log`
3. Load and preprocess data
4. Train the model with your chosen method
5. Save best model and generate plots
6. Track health metrics (if confidence_aware)

### Training Methods

#### 1. Standard Multiclass
```json
{
    "training_method": "standard_multiclass",
    "loss_function": "CrossEntropyLoss"
}
```

#### 2. Confidence-Aware (Curriculum Learning)
```json
{
    "training_method": "confidence_aware",
    "loss_function": "CrossEntropyLoss",
    "rank_weight": 0.5,
    "rank_weight_f": 0.5
}
```
Tracks sample-level metrics:
- **AUM (Area Under Margin)**: Measures model confidence on samples
- **EL2N (Error L2 Norm)**: Tracks prediction errors
- **Forgetting Events**: Identifies samples learned then forgotten

#### 3. Focal Loss for Imbalanced Data
```json
{
    "training_method": "standard_multiclass",
    "loss_function": "FocalLoss",
    "focal_loss_config": {"gamma": 2.0, "alpha": [0.25, 0.75]}
}
```

## 📊 Output Files

### Models
```
Models/{exp_name}/{model_name}/{model_name}.pth
```
Contains: `{'model_state_dict': state_dict}`

### Logs
```
logs/{exp_name}_20260211_143025.log
```
Contains: All training info, errors with full tracebacks

### Training Plots
```
plots/{exp_name}/{model_name}/loss_curve.png
```

### History Metrics (confidence_aware only)
```� Package Modules

### `nn_tab/datasets/`
**Data loading, preprocessing, and dataset management**

#### `dataset.py`
- `read_train_data()`: Main data loading function that handles train/val split, normalization, and DataLoader creation
- Supports parquet and CSV file formats
- Automatic feature extraction and target handling

# `model.py`
- `fraudmodel_3layer`: 3-layer fully connected network
- `fraudmodel_5layer`: 5-layer fully connected network (default)
- `fraudmodel_7layer`: 7-layer fully connected network
- `fraudmodel_8layer`: 8-layer fully connected network
- All models support dynamic hidden layer configuration

#### `model_initialise.py`
- `get_model()`: Main function to initialize model, loss, optimizer, and scheduler
- `FocalLoss`: Implementation of Focal Loss for handling class imbalance
- `_create_loss_function()`: Factory function for loss function creation
- Supports multiple optimizers: Adam, SGD, AdamW, RMSprop, Adamax
- ReduceLROnPlateau scheduler with patience=4

### `nn_tab/utils/`
**Training algorithms, evaluation, and metrics**

#### `utils.py`
- `train_model()`: Standard training loop with early stopping
- `train_model_crl()`: Confidence-aware training with curriculum learning
- `train_fn()`: Single epoch training for standard methods
- `train_crl()`: Single epoch training for confidence-aware method with ranking losses
- `val_fn()`: Validation evaluation with metrics computation
- `accuracy_metric()`: Accuracy calculation utility
- `check_for_invalid_values()`: NaN/Inf detection in tensors
- `AverageMeter`: Utility class for tracking running averages

#### `calculate_scores.py`
- `calculate_aum()`: Computes Area Under Margin (AUM) scores per sample
- `EL2N_score()`: Calculates Error L2 Norm for sample difficulty
- `update_forgetting()`: Tracks forgetting events per sample
- `add_logits_to_aum_dict()`: Adds model predictions to AUM dictionary
- Used exclusively in confidence_aware training for sample health tracking

Add new architectures in `nn_tab/models/model.py`:
```python
class fraudmodel_custom(nn.Module):
    def __init__(self, input_dim, hidden_layers, output):
        super().__init__()
        # Define your layers
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        # ... more layers
        self.output = nn.Linear(hidden_layers[-1], output)
```

Then register in `model_initialise.py`:
```python
model_mapping = {
    'fraudmodel_5layer': fraudmodel_5layer,
    'fraudmodel_custom': fraudmodel_custom,  # Add here
}
```

### Adding Custom Loss Functions
Extend `_create_loss_function()` in `nn_tab/models/model_initialise.py`:
```python
elif loss_name == "MyCustomLoss":
    criterion = MyCustomLoss().to(device)
    logger.info("Using MyCustomLoss")
```

### Adding Custom Metrics
Add to `nn_tab/utils/calculate_scores.py`:
```python
def my_custom_score(logits, targets, sample_ids, score_dict, epoch):
    """Calculate custom metric per sample."""
    # Your implementation
    return score_dict
```

| Issue | Solution |
|-------|----------|
| Training method not recognized | Ensure `training_method` is `"confidence_aware"`, `"standard_multiclass"`, or `"standard_multilabel"` |
| Out of memory errors | Reduce `batch_size` or use smaller `hidden_layers` |
| NaN/Inf losses | Lower learning rate, check input data for invalid values, or add gradient clipping |
| Slow training | Increase `batch_size`, reduce model size, or set `num_workers > 0` |
| Model not improving | Try different loss function, adjust learning rate, or check class weights |
| File not found errors | Verify all paths in config.json are absolute or relative to execution directory |files in `logs/{exp_name}_{timestamp}.log`
- Captures all INFO, WARNING, and ERROR messages

## 🎓 Training Methods

### 1. Standard Multiclass
Traditional supervised learning for multi-class classification.

**Configuration:**
```json
{
    "training_method": "standard_multiclass",
    "loss_function": "CrossEntropyLoss"
}
```

**Use When:**
- Balanced or moderately imbalanced datasets
- Standard classification tasks
- Fast training needed

**Loss Function:** Classification loss only

### 2. Standard Multilabel
Multi-label classification where samples can belong to multiple classes.
Version**: 1.0.0  
**Last Updated**: February 2026
```json
{
    "training_method": "standard_multilabel",
    "loss_function": "BCEWithLogitsLoss"
}
```

**Use When:**
- Samples have multiple labels
- Non-exclusive categories
- Tag prediction tasks

**Loss Function:** Binary cross-entropy with logits

### 3. Confidence-Aware (Curriculum Learning)
Advanced training method that uses sample difficulty scores and ranking losses.

**Configuration:**
```json
{
    "training_method": "confidence_aware",
    "loss_function": "CrossEntropyLoss",
    "rank_weight": 0.5,
    "rank_weight_f": 0.5
}
```

**Use When:**
- Noisy labels suspected
- Want sample-level health metrics
- Need to identify hard/easy samples
- Data quality analysis required

**Loss Function:**
```
Total Loss = Classification Loss + 
             rank_weight × Correctness Ranking Loss + 
             rank_weight_f × Forgetting Ranking Loss
```

**Outputs:**
- AUM scores: Sample margin trends over epochs
- EL2N scores: Prediction error magnitudes
- Forgetting scores: Learning stability per sample

**How It Works:**
1. Tracks correctness history for each sample
2. Monitors forgetting events (learned → forgotten)
3. Applies ranking loss to encourage higher confidence on easy samples
4. Adapts focus to harder samples as training progresses

## 🔧 Available Loss Functions

### CrossEntropyLoss
**Best for:** Balanced multiclass classification

**Configuration:**
```json
{
    "loss_function": "CrossEntropyLoss",
    "use_class_weights": true
}
```

**Features:**
- Supports automatic class weighting
- Standard for multiclass problems
- Numerically stable

### FocalLoss
**Best for:** Heavily imbalanced datasets

**Configuration:**
```json
{
    "loss_function": "FocalLoss",
    "focal_loss_config": {"gamma": 2.0, "alpha": [0.25, 0.75]},
    "use_class_weights": false
}
```

**Parameters:**
- `gamma`: Focusing parameter (default: 2.0) - higher values focus more on hard examples
- `alpha`: Class balancing weights (optional) - list of per-class weights

**Formula:** `FL(pt) = -α(1-pt)^γ * log(pt)`

### BCEWithLogitsLoss
**Best for:** Multilabel classification

**Configuration:**
```json
{
    "loss_function": "BCEWithLogitsLoss",
    "training_method": "standard_multilabel"
}
```

**Features:**
- Combines sigmoid + BCE for numerical stability
- Supports pos_weight for class imbalance
- Used for multi-label problems

### MSELoss
**Best for:** Regression tasks or soft labels

**Configuration:**
```json
{
    "loss_function": "MSELoss"
}
```

**Features:**
- Mean squared error loss
- Suitable for continuous targets
- Can be used with soft labelsg as dict keys
4. **Directory Creation**: Moved history metrics folder creation outside training loop (100x efficiency gain)

#### Enhancements
1. **Experiment Organization**: All outputs now organized by `exp_name` for clean multi-experiment workflows
2. **Exception Handling**: Global exception hook captures and logs all runtime errors with full tracebacks
3. **Defensive Programming**: Safe tensor/list conversions throughout confidence-aware pipeline
4. **Path Management**: Consistent use of `os.path.join()` for cross-platform compatibility

## 🔍 Confidence-Aware Training Details

Confidence-aware training uses curriculum learning principles:

1. **Correctness History**: Tracks how often each sample is classified correctly
2. **Forgetting Tracking**: Monitors samples that were learned then forgotten
3. **Ranking Loss**: Encourages model to be more confident on easy samples than hard ones
4. **Adaptive Learning**: Focuses on harder samples as training progresses

### Loss Formulation
```
Total Loss = Classification Loss + 
             rank_weight × Correctness Ranking Loss + 
             rank_weight_f × Forgetting Ranking Loss
```

## 📈 Monitoring Training

### Log File
All training events are logged with timestamps:
```
2026-02-11 14:30:25 - nntab - INFO - Training started!!
2026-02-11 14:30:30 - nntab - INFO - Epoch=1, Train Loss=0.6234, Test Loss=0.5891, test aucpr=0.7234, Test Accuracy=82.45
```

### Real-time Monitoring
Progress bars show:
- Training batches per epoch
- Validation batches per epoch
- Loss and accuracy metrics

## 🛠️ Advanced Usage

### Custom Model Architecture
Edit `nn_tab/models/model.py` to add new architectures:
```python
class MyCustomModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, output):
        # Your architecture
```

### Custom Loss Functions
Add to `nn_tab/models/model_initialise.py`:
```python
def _create_loss_function(loss_name, ...):
    # Add your custom loss
```

### Custom Metrics
Add to `nn_tab/utils/calculate_scores.py`:
```python
def my_custom_metric(logits, targets, sample_ids, ...):
    # Your metric calculation
```

## 🐞 Troubleshooting

### Issue: Training method not recognized
**Solution**: Ensure `training_method` in config.json is one of: `"confidence_aware"`, `"standard_multiclass"`, or `"standard_multilabel"`

### Issue: KeyError with sample_ids
**Solution**: Updated in latest version - defensive tensor conversion now handles this automatically

### Issue: Out of memory
**Solution**: Reduce `batch_size` in config.json

### Issue: NaN losses
**Solution**: 
- Reduce learning rate in optimizer config
- Check for inf/NaN in input data
- Enable gradient clipping

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For issues and questions, please open a GitHub issue.

---

**Last Updated**: February 11, 2026  
**Version**: 0.2.0
