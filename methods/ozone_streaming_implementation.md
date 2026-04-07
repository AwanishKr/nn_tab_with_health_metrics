# Ozone Partition Streaming — Implementation Approach

## Problem

The existing training code (`train_model_crl`) assumes the entire training set fits in RAM as a single DataLoader. For large datasets on Apache Ozone, this is not feasible. The constraint is:

- **At most 2 partitions in RAM** at any time
- **History / ForgettingTracker / TrainingSignalCollector** arrays must be sized by the total number of training samples upfront
- **Eval pass** (to record per-sample signals) must happen on each partition *while it is still in RAM*, at the model state immediately after training on that partition
- **No changes** to existing `train_crl`, `val_fn`, `History`, `ForgettingTracker`, `TrainingSignalCollector`

---

## Why a New Training Function is Necessary

The core issue is **loop structure**, not data loading.

`train_model_crl` has this shape per epoch:

```
for batch in train_loader:    ← consumes all data
    forward / backward

eval_pass(train_loader, ...)  ← iterates again — data still in RAM (small dataset)
```

For Ozone, data cannot all be in RAM at once. The required shape is:

```
for partition in streamer:
    train_crl(part_loader, ...)   ← train on partition
    eval_pass(part_loader, ...)   ← eval while partition is STILL in RAM
    del part_loader               ← free partition
```

The eval pass must fire per partition at the model state *right after training on that partition*. Any approach that delays the eval pass (e.g. doing it at epoch end) uses a stale model state and produces wrong metrics.

This boundary ownership — holding the model, the data, and the eval logic together at the partition level — cannot be expressed inside a DataLoader or IterableDataset. It requires an explicit training loop.

---

## Why Not an IterableDataset

An `IterableDataset.__iter__` yields individual samples. The DataLoader drives the iteration via `next()`, accumulating batches transparently. Partition boundaries are invisible to the training loop:

```
DataLoader calls next() repeatedly
  → generator loads partition A, yields samples one by one
  → partition A exhausted, generator loads partition B automatically
  → training loop sees one flat stream of batches, no boundary signal
```

To fire an eval pass at a partition boundary, the iterator would need to hold a reference to the model, criterion, collector, device, and epoch number — turning the data layer into a training loop. That is the wrong abstraction.

---

## Design

### PartitionDataset

Wraps one in-memory partition DataFrame as a standard `torch.utils.data.Dataset`.

```python
__getitem__ → (x, y, sample_id, sample_id)
```

This tuple is **identical** to `Custom_Dataset_CRL`, so `train_crl` and `val_fn` work with zero changes.

For the validation set (no `sample_id` column), passing `sample_id_col=None` falls back to a sequential dummy index — `val_fn` ignores it anyway.

### PartitionStreamer

Yields whole DataFrames (not samples) to the training loop. A background prefetch thread loads the next partition into a `queue.Queue(maxsize=1)` while the current one is being consumed.

```
queue maxsize=1 → at most 2 partitions in RAM simultaneously
backpressure is automatic → producer blocks if consumer is slow
```

`stream()` is a generator. It restarts the prefetch thread fresh on each call (once per epoch).

### Footer scan

`count_total_rows` calls `pq.read_metadata()` per partition — reads only metadata from the Parquet footer, no row data loaded. Used once at startup to get `N_train` for sizing the tracker arrays.

### Validation set

Loaded once into RAM at startup as a `PartitionDataset` with `sample_id_col=None`. Reused every epoch via a persistent `val_loader`. Never re-downloaded.

---

## Training Loop Structure

```
read_train_data_ozone()
  footer scan → total_rows N
  build PartitionStreamer
  load val set once → val_loader

train_model_crl_ozone()
  init History(N), ForgettingTracker(N), TrainingSignalCollector(N)

  for epoch in 1..epochs:
      epoch_seed = epoch * 1000

      for partition_df in streamer.stream():
          partition_df.sample(frac=1, random_state=epoch_seed)   # row shuffle
          part_ds = PartitionDataset(partition_df, ...)
          del partition_df                                         # raw df freed
          part_loader = DataLoader(part_ds, ...)

          train_crl(part_loader, model, ...)                       # training pass
          eval pass on part_loader (model.eval(), no_grad)         # eval pass — partition still in RAM
          collector.compute_grad_norms(model, part_loader, ...)
          del part_ds, part_loader                                 # partition freed

      val_fn(model, val_loader, ...)
      early stopping / model saving

  # Post-training: two more streaming passes for embedding extraction
  #   pass 1: penultimate embeddings via forward hook on last Linear layer
  #   pass 2: intermediate embeddings via hooks at 25 / 50 / 75% depth

  collector.save_raw_arrays(...)
  plot_loss_curve(...)
```

---

## Files Changed

| File | Change |
|------|--------|
| `nn_tab/datasets/ozone_dataset.py` | Created — `PartitionDataset`, `PartitionStreamer`, `read_train_data_ozone`, discovery & footer scan helpers |
| `nn_tab/utils/utils.py` | Appended `train_model_crl_ozone` after `train_model` |
| `nn_tab/utils/__init__.py` | Added `train_model_crl_ozone` to exports |
| `nn_tab/datasets/__init__.py` | Added all 5 Ozone symbols to exports |
| `main.py` | Added Ozone branch — builds `fs`, calls `read_train_data_ozone`, dispatches to `train_model_crl_ozone` |
| `config.json` | Added 7 Ozone keys (empty by default — local path used when unset) |
| `requirements.txt` | Added `pyarrow>=13.0.0` |

---

## Activating Ozone Mode

In `config.json`, set the two path fields:

```json
"ozone_train_path": "bucket/path/to/train/partitions",
"ozone_val_path":   "bucket/path/to/val.parquet",
"ozone_fs_mode":    "s3",
"ozone_s3_endpoint": "http://ozone-s3g-host:9878",
"ozone_aws_access_key": "...",
"ozone_aws_secret_key": "..."
```

When both paths are empty strings (default), the code falls back to the original local `read_train_data` path unchanged.
