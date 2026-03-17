import os
import json
import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.neighbors import NearestNeighbors



def calculate_aum(logits, targets, sample_ids, aum_dict, epoch):
    target_values = logits.gather(1, targets.view(-1, 1)).squeeze()

    # mask out target values
    masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float('-inf'))
    other_logit_values, _ = masked_logits.max(1)
    other_logit_values = other_logit_values.squeeze()
    margin_values = (target_values - other_logit_values).tolist()

    for sample_id, margin in zip(sample_ids, margin_values):
        if sample_id not in aum_dict:
            aum_dict[sample_id] = {
                "epoch": epoch,
                "margin_sum": margin,
                "aum": margin
            }
        if "epoch" not in aum_dict[sample_id]:  # first time seeing this sample
            aum_dict[sample_id] = {
                "epoch": epoch,
                "margin_sum": margin,
                "aum": margin
            }
        else:
            aum_dict[sample_id]["epoch"] = epoch
            aum_dict[sample_id]["margin_sum"] += margin
            aum_dict[sample_id]["aum"] = (
                aum_dict[sample_id]["margin_sum"] / aum_dict[sample_id]["epoch"]
            )

    return aum_dict


# def compute_grand_score(model, X_batch, y_batch, criterion, grand_scores, sample_ids):
#     """
#     GRAND computation:
#     - uses per-sample loss (criterion with reduction='none')
#     - computes gradients with torch.autograd.grad
#     - maintains {id: [count, sum, mean]} dict
#     """
#     model.eval()  # disable dropout/bn randomness

#     X_batch, y_batch = X_batch.to(next(model.parameters()).device), y_batch.to(next(model.parameters()).device)

#     # Ensure gradients are enabled
#     with torch.set_grad_enabled(True):
#         outputs = model(X_batch)
#         losses = criterion(outputs, y_batch)  # shape [B], still connected to graph

#         grad_norms = []
#         for i in range(X_batch.size(0)):
#             grads = torch.autograd.grad(
#                 losses[i],
#                 model.parameters(),
#                 retain_graph=True,
#                 create_graph=False
#             )
#             grad_norm = torch.cat([g.reshape(-1) for g in grads]).norm().item()
#             grad_norms.append(grad_norm)

#     # update dict {id: [count, sum, mean]}
#     for sample_id, gn in zip(sample_ids, grad_norms):
#         if len(grand_scores[sample_id]) == 0:
#             grand_scores[sample_id] = {
#                 "count": 1,
#                 "gn_sum": gn,
#                 "gn_mean": gn
#             }
#             # grand_scores[sample_id].extend([1, gn, gn])  # count, sum, mean
#         else:
#             grand_scores[sample_id]["count"] += 1
#             grand_scores[sample_id]["gn_sum"] += gn
#             grand_scores[sample_id]["gn_mean"] = grand_scores[sample_id][1] / grand_scores[sample_id][0]
            

#     return grand_scores


            
    
def EL2N_score(probs, y, sample_ids, el2n_scores, epoch):
    """
    Compute/update EL2N scores.
    
    probs: [B, C] softmax probabilities
    y:     [B] ground-truth labels
    sample_ids: list/array of IDs for the batch
    el2n_scores: dict {id: [count, sum, mean]} to be updated
    """
    one_hot = torch.nn.functional.one_hot(y, num_classes=probs.size(1)).float()
    errors = torch.norm(probs - one_hot, dim=1)   # [B]

    for sid, err in zip(sample_ids, errors):
        err_val = err.item()
        if sid not in el2n_scores:
            el2n_scores[sid] = {
                "epoch": epoch,
                "sum": err_val,
                "mean": err_val,
            }
        elif len(el2n_scores[sid]) == 0:
            el2n_scores[sid].extend([epoch, err_val, err_val])  # count, sum, mean
        else:
            el2n_scores[sid]['epoch'] = epoch
            el2n_scores[sid]['sum'] += err_val
            el2n_scores[sid]['mean'] = el2n_scores[sid]['sum'] / el2n_scores[sid]['epoch']

    return el2n_scores
        
            
            
def update_forgetting(logits, labels, sample_ids, epoch, forgetting_scores):
    """
    Update forgetting statistics.

    forgetting_scores[sid] = {
        "forget_count": int,
        "first_learned": int or None,
        "last_learn_epoch": int or None,
        "correct_count": int,
        "first_forget_epoch": int or None
    }
    """
    preds = logits.argmax(dim=1)

    for pred, label, sid in zip(preds, labels, sample_ids):
        # Initialize entry if first time
        if sid not in forgetting_scores:
            forgetting_scores[sid] = {
                "forget_count": 0,
                "first_learned": None,
                "last_learn_epoch": None,
                "correct_count": 0,
                "first_forget_epoch": None,
                "_last_correct": False  # internal tracker, will be deleted later
            }

        entry = forgetting_scores[sid]
        was_correct = entry["_last_correct"]
        now_correct = (pred.item() == label.item())

        # Count correct predictions
        if now_correct:
            entry["correct_count"] += 1
            if entry["first_learned"] is None:
                entry["first_learned"] = epoch
            if not was_correct:  # flip from wrong → correct
                entry["last_learn_epoch"] = epoch

        # Forgetting event
        if was_correct and not now_correct:
            entry["forget_count"] += 1
            if entry["first_forget_epoch"] is None:
                entry["first_forget_epoch"] = epoch

        # Update last correctness
        entry["_last_correct"] = now_correct

    return forgetting_scores





def compute_gradient_norms_pass(model, loader, device, grad_norm_array, epoch_idx, criterion):
    """
    One pass over the training loader to compute per-sample gradient norms (GraNd signal).
    Fills grad_norm_array[:, epoch_idx] in-place.

    Only intended for the first few epochs (e.g. epoch_idx < 5).
    loader must yield (X, y, idx, identifier) batches.

    Args:
        model: PyTorch model
        loader: DataLoader yielding (X, y, idx, identifier)
        device: torch device
        grad_norm_array: np.ndarray of shape [N_samples, K_epochs], filled in-place
        epoch_idx: column index (0-based) to write into grad_norm_array
        criterion: loss function with reduction='none' (matches the configured training loss)
    """
    model.eval()
    last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]

    for X, y, idx, _ in loader:
        X = X.to(device, dtype=torch.float32)
        y = y.to(device)
        batch_size = X.size(0)

        model.zero_grad()
        outputs = model(X)
        losses = criterion(outputs, y)  # [B]

        for i in range(batch_size):
            grads = torch.autograd.grad(
                losses[i],
                last_linear.parameters(),
                retain_graph=(i < batch_size - 1),
                create_graph=False
            )
            grad_norm = torch.cat([g.reshape(-1) for g in grads]).norm().item()
            grad_norm_array[idx[i].item(), epoch_idx] = grad_norm

    model.train()


def prediction_depth_knn(layer_reps, y, sample_ids, depth_scores, k=10):
    """
    layer_reps: dict[layer_idx -> tensor [N, D]] representations
    y: [N] ground-truth labels (long tensor)
    sample_ids: list of IDs
    depth_scores: dict {id: [depth]} to be updated
    k: neighbors
    
    Returns: updated depth_scores
    """
    N = y.size(0)
    y_np = y.cpu().numpy()

    # Init
    for sid in sample_ids:
        if sid not in depth_scores:
            depth_scores[sid] = [float('inf')]

    for layer_idx, X in layer_reps.items():
        X_np = X.detach().cpu().numpy()

        nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X_np)
        _, neigh_idx = nn.kneighbors(X_np)

        for i, sid in enumerate(sample_ids):
            neighbors = neigh_idx[i][1:]  # exclude self
            maj_vote = np.bincount(y_np[neighbors]).argmax()
            if depth_scores[sid][0] == float('inf') and maj_vote == y_np[i]:
                depth_scores[sid][0] = layer_idx

    return depth_scores



class TrainingSignalCollector:
    """Collects raw per-sample training signals for DataGenome analysis.

    Captures only the minimal raw signals during training: logits, per-sample
    loss, correctness flags, and gradient norms. All derived metrics (AUM,
    EL2N, forgetting events, etc.) are computed post-training from these arrays.
    """

    def __init__(self, num_samples, epochs, num_classes, criterion_cls):
        self.logits_array = np.zeros((num_samples, epochs, num_classes), dtype=np.float16)
        self.loss_array = np.zeros((num_samples, epochs), dtype=np.float16)
        self.correct_array = np.zeros((num_samples, epochs), dtype=np.uint8)
        self.targets_array = np.full(num_samples, -1, dtype=np.int64)
        self.grad_norm_epochs = min(5, epochs)
        self.grad_norm_array = np.zeros((num_samples, self.grad_norm_epochs), dtype=np.float16)

        self._per_sample_criterion = deepcopy(criterion_cls)
        self._per_sample_criterion.reduction = 'none'


    def update_batch(self, output, target, idx, epoch_num):
        """Record raw per-sample signals for one batch.

        Args:
            output: model logits [B, C] (detached or not, will be detached here).
            target: ground-truth labels [B].
            idx: sample indices [B] mapping into the pre-allocated arrays.
            epoch_num: 1-based epoch number.
        """
        idx_np = idx.cpu().numpy()
        ep = epoch_num - 1  # 0-based column index

        self.logits_array[idx_np, ep, :] = output.detach().cpu().numpy()
        self.targets_array[idx_np] = target.detach().cpu().numpy()

        per_sample_loss = self._per_sample_criterion(output.detach(), target.detach())
        self.loss_array[idx_np, ep] = per_sample_loss.cpu().numpy()

        _, predicted = output.detach().max(1)
        self.correct_array[idx_np, ep] = (predicted == target).cpu().numpy().astype(np.uint8)


    def compute_grad_norms(self, model, loader, device, epoch_num):
        """Compute gradient norms for the first grad_norm_epochs epochs."""
        if epoch_num <= self.grad_norm_epochs:
            compute_gradient_norms_pass(model, loader, device, self.grad_norm_array, epoch_num - 1, self._per_sample_criterion)


    def extract_embeddings(self, model, loader, device):
        """Extract penultimate-layer embeddings via a single eval pass.

        Registers a forward hook on the last Linear layer to capture its input
        (the penultimate activation). Stores result in self.embeddings_array [N, D].
        """
        last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
        embed_dim = last_linear.in_features
        num_samples = self.logits_array.shape[0]
        self.embeddings_array = np.zeros((num_samples, embed_dim), dtype=np.float16)

        captured = {}

        def hook_fn(module, input, output):
            captured['input'] = input[0].detach()

        handle = last_linear.register_forward_hook(hook_fn)

        model.eval()
        with torch.no_grad():
            for X, _, idx, _ in loader:
                X = X.to(device, dtype=torch.float32)
                model(X)
                self.embeddings_array[idx.numpy()] = captured['input'].cpu().numpy()

        handle.remove()
        model.train()

    def extract_intermediate_embeddings(self, model, loader, device):
        """Extract embeddings at 25%, 50%, 75% network depth for prediction depth probing.

        For each depth level, hooks the corresponding Linear layer to capture its input
        (the representation at that point in the network). Stores results in
        self.intermediate_embeddings: dict mapping depth label to [N, D] array.
        """
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        # Exclude the output (last) layer — hidden layers only
        hidden_linears = linear_layers[:-1]
        num_hidden = len(hidden_linears)

        if num_hidden < 2:
            # Too shallow for meaningful intermediate probing; skip
            self.intermediate_embeddings = {}
            return

        # Select layers at approximately 25%, 50%, 75% depth
        depth_targets = {
            'depth_25': max(0, round(num_hidden * 0.25) - 1),
            'depth_50': max(0, round(num_hidden * 0.50) - 1),
            'depth_75': max(0, round(num_hidden * 0.75) - 1),
        }

        # Deduplicate if indices collide (e.g. very shallow network)
        selected = {}
        for label, idx in depth_targets.items():
            selected[label] = hidden_linears[idx]

        num_samples = self.logits_array.shape[0]
        captured = {}
        handles = []

        for label, layer in selected.items():
            embed_dim = layer.in_features
            captured[label] = {
                'array': np.zeros((num_samples, embed_dim), dtype=np.float16),
                'batch_input': None,
            }

            def make_hook(lbl):
                def hook_fn(module, inp, out):
                    captured[lbl]['batch_input'] = inp[0].detach()
                return hook_fn

            handles.append(layer.register_forward_hook(make_hook(label)))

        model.eval()
        with torch.no_grad():
            for X, _, idx, _ in loader:
                X = X.to(device, dtype=torch.float32)
                model(X)
                idx_np = idx.numpy()
                for label in selected:
                    captured[label]['array'][idx_np] = (
                        captured[label]['batch_input'].cpu().numpy()
                    )

        for h in handles:
            h.remove()
        model.train()

        self.intermediate_embeddings = {
            label: captured[label]['array'] for label in selected
        }

    def save_raw_arrays(self, raw_signals_dir, actual_epochs=None, configured_epochs=None, early_stopped=False):
        """Save all raw signal arrays as .npy files and training metadata as JSON."""
        # Trim zero-padded epoch columns if early stopping occurred
        if actual_epochs is not None and actual_epochs < self.logits_array.shape[1]:
            self.logits_array = self.logits_array[:, :actual_epochs, :]
            self.loss_array = self.loss_array[:, :actual_epochs]
            self.correct_array = self.correct_array[:, :actual_epochs]

        np.save(os.path.join(raw_signals_dir, "logits_array.npy"), self.logits_array)
        np.save(os.path.join(raw_signals_dir, "loss_array.npy"), self.loss_array)
        np.save(os.path.join(raw_signals_dir, "correct_array.npy"), self.correct_array)
        np.save(os.path.join(raw_signals_dir, "grad_norm_array.npy"), self.grad_norm_array)

        if hasattr(self, 'embeddings_array'):
            np.save(os.path.join(raw_signals_dir, "embeddings.npy"), self.embeddings_array)

        if hasattr(self, 'intermediate_embeddings') and self.intermediate_embeddings:
            for label, arr in self.intermediate_embeddings.items():
                np.save(os.path.join(raw_signals_dir, f"embeddings_{label}.npy"), arr)

        if actual_epochs is not None:
            metadata = {
                "actual_epochs": actual_epochs,
                "configured_epochs": configured_epochs,
                "early_stopped": early_stopped,
                "num_samples": int(self.logits_array.shape[0]),
                "num_classes": int(self.logits_array.shape[2]),
                "grad_norm_epochs": int(self.grad_norm_epochs),
                "intermediate_embedding_depths": list(self.intermediate_embeddings.keys()) if hasattr(self, 'intermediate_embeddings') and self.intermediate_embeddings else [],
            }
            with open(os.path.join(raw_signals_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)


    def save_dataset_snapshot(self, train_loader, raw_signals_dir):
        """Save full training dataset (features + idx + true_label) as parquet.

        idx is positional (0..N-1) assigned at dataset creation before training starts.
        This snapshot lets post-training analysis join any row of the raw signal arrays
        back to actual feature values and labels without re-loading the original data.
        """
        dataset = train_loader.dataset
        N = len(dataset)
        if hasattr(dataset, 'feature_columns'):
            df = pd.DataFrame(dataset.x_data, columns=dataset.feature_columns)
        else:
            df = pd.DataFrame(dataset.x_data, columns=[f'feature_{i}' for i in range(dataset.x_data.shape[1])])
        df.insert(0, 'idx', np.arange(N))
        df['true_label'] = dataset.y_data
        df.to_parquet(os.path.join(raw_signals_dir, "training_samples.parquet"), index=False)


# import faiss
# import numpy as np
# import torch


# def prediction_depth_knn_faiss(layer_storage, labels, sample_ids, k=10):
#     """
#     For each sample, finds the first layer where kNN agrees with the label.
#     Returns dict mapping sample_id -> depth.
#     """
#     depth_scores = {sid: None for sid in sample_ids}
#     labels = labels.numpy()

#     for layer_idx, X in layer_storage.items():
#         X = X.numpy().astype("float32")

#         # FAISS index
#         index = faiss.IndexFlatL2(X.shape[1])
#         index.add(X)

#         D, I = index.search(X, k + 1)  # include self
#         knn_labels = labels[I[:, 1:]]  # skip self

#         # majority vote
#         preds = np.array([np.argmax(np.bincount(neigh)) for neigh in knn_labels])

#         for i, sid in enumerate(sample_ids):
#             if depth_scores[sid] is None and preds[i] == labels[i]:
#                 depth_scores[sid] = layer_idx

#     # fill unresolved with max depth
#     for sid in sample_ids:
#         if depth_scores[sid] is None:
#             depth_scores[sid] = max(layer_storage.keys())

#     return depth_scores


# import numpy as np
# import torch

# def prediction_depth_knn_faiss(layer_storage, labels, sample_ids, k=10):
#     """
#     For each sample, finds the first layer where kNN agrees with the label.
#     - Uses FAISS GPU if available
#     - Falls back to FAISS CPU if no GPU
#     - Falls back to sklearn if FAISS not installed

#     Args:
#         layer_storage: dict[layer_idx -> torch.Tensor [N, D]]
#         labels: torch.Tensor [N]
#         sample_ids: list of IDs
#         k: number of neighbors

#     Returns:
#         depth_scores: dict {sample_id -> depth}
#     """
#     depth_scores = {sid: None for sid in sample_ids}
#     labels = labels.cpu().numpy()

#     # Try FAISS first
#     try:
#         import faiss
#         use_gpu = faiss.get_num_gpus() > 0
#         if use_gpu:
#             res = faiss.StandardGpuResources()
#     except ImportError:
#         faiss = None
#         use_gpu = False

#     for layer_idx, X in layer_storage.items():
#         X = X.detach().cpu().numpy().astype("float32")

#         if faiss is not None:
#             d = X.shape[1]
#             index_flat = faiss.IndexFlatL2(d)

#             if use_gpu:
#                 index = faiss.index_cpu_to_gpu(res, 0, index_flat)
#             else:
#                 index = index_flat

#             index.add(X)
#             D, I = index.search(X, k + 1)  # include self
#         else:
#             # sklearn fallback
#             from sklearn.neighbors import NearestNeighbors
#             nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
#             D, I = nn.kneighbors(X)

#         knn_labels = labels[I[:, 1:]]  # skip self
#         preds = np.array([np.argmax(np.bincount(neigh)) for neigh in knn_labels])

#         for i, sid in enumerate(sample_ids):
#             if depth_scores[sid] is None and preds[i] == labels[i]:
#                 depth_scores[sid] = layer_idx

#     # fill unresolved with max depth
#     max_layer = max(layer_storage.keys())
#     for sid in sample_ids:
#         if depth_scores[sid] is None:
#             depth_scores[sid] = max_layer

#     return depth_scores

