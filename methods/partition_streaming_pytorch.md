# Partition Streaming Layer for PyTorch — Large Parquet Dataset on Apache Ozone

## Context & Problem

An existing Python package handles:
- Dataset loading (previously fit entirely in RAM)
- Batch generation
- History-based trackers for **confidence-aware loss** and **forgetting-aware loss** functions

The dataset has now grown too large to load into RAM at once. It is stored as **partitioned Parquet files** inside an **Apache Ozone** directory (accessed via Ozone's S3-compatible gateway using PyArrow).

**Goal:** Stream one partition at a time into the existing package without modifying its internals. While one partition is being trained on, the next partition is prefetched from Ozone in a background thread. Once training on a partition finishes, it is dropped from RAM.

---

## Architecture

```
Apache Ozone (Parquet folder)
         │
         ▼
 PartitionStreamer                   ← NEW: streams partitions, prefetches in background
  ├── active partition (in RAM)      ← currently training
  └── prefetch partition (loading)   ← background thread loading next partition
         │
         ▼
 PartitionDataset                    ← NEW: wraps one partition as a standard PyTorch Dataset
         │
         ▼
 Existing Package (unchanged)        ← receives Dataset, creates batches,
  ├── DataLoader creation                 updates confidence + forgetting trackers
  ├── confidence tracker
  └── forgetting tracker
         │
         ▼
  Neural Network Training Loop
```

### Memory guarantee
At most **2 partitions** are in RAM at any time: the one being trained on, and the one being prefetched. The prefetch thread blocks automatically (backpressure) if training hasn't finished the current partition yet.

---

## Full Implementation

### Dependencies

```bash
pip install torch pyarrow pandas
```

---

### `partition_streamer.py`

```python
import threading
import queue
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PartitionStreamer:
    """
    Streams Parquet partitions from Apache Ozone one at a time.
    Prefetches the next partition in a background thread while
    the current partition is being consumed by the training loop.

    Memory guarantee: at most 2 partitions in RAM simultaneously.
    Backpressure is automatic via queue maxsize=1.
    """

    def __init__(
        self,
        fs,
        partition_paths: list[str],
        feature_cols: list[str],
        label_col: str
    ):
        """
        Args:
            fs              : PyArrow S3FileSystem pointed at Ozone S3 gateway
            partition_paths : Ordered list of parquet file paths on Ozone
            feature_cols    : List of feature column names
            label_col       : Name of the label/target column
        """
        self.fs = fs
        self.partition_paths = partition_paths
        self.feature_cols = feature_cols
        self.label_col = label_col
        self._prefetch_queue = queue.Queue(maxsize=1)  # 1 partition buffered ahead
        self._stop_event = threading.Event()

    def _load_partition(self, path: str) -> pd.DataFrame:
        table = pq.read_table(
            path,
            filesystem=self.fs,
            columns=self.feature_cols + [self.label_col]
        )
        return table.to_pandas()

    def _prefetch_worker(self, paths: list[str]):
        for path in paths:
            if self._stop_event.is_set():
                break
            df = self._load_partition(path)
            self._prefetch_queue.put(df)   # blocks if queue full — backpressure
        self._prefetch_queue.put(None)     # sentinel: signals all partitions done

    def stream(self):
        """
        Generator that yields one partition DataFrame at a time.
        The previous partition is released from RAM before the next is yielded.
        """
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(self.partition_paths,),
            daemon=True
        )
        thread.start()

        while True:
            partition_df = self._prefetch_queue.get()  # blocks until next is ready
            if partition_df is None:
                break
            yield partition_df
            # partition_df goes out of scope here → eligible for GC

        self._stop_event.set()
        thread.join()


class PartitionDataset(Dataset):
    """
    Wraps a single in-memory partition (Pandas DataFrame) as a
    standard PyTorch Dataset. This is what gets passed to the
    existing package for DataLoader creation and tracker updates.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], label_col: str):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

---

### `train.py` — Integration with existing package

```python
from partition_streamer import PartitionStreamer, PartitionDataset
import pyarrow.fs as pafs


def train(
    model,
    optimizer,
    criterion,
    your_package,       # existing package instance — untouched
    fs,
    partition_paths,
    feature_cols,
    label_col
):
    streamer = PartitionStreamer(fs, partition_paths, feature_cols, label_col)

    for partition_idx, partition_df in enumerate(streamer.stream()):

        # Wrap partition as a standard PyTorch Dataset
        dataset = PartitionDataset(partition_df, feature_cols, label_col)

        # Pass to existing package — it creates batches and updates trackers as before
        loader = your_package.create_dataloader(dataset)
        your_package.update_trackers(partition_idx, dataset)

        # Explicitly drop the raw DataFrame — tensors in dataset are sufficient now
        del partition_df

        # Training loop on this partition
        model.train()
        for features, labels in loader:
            features = features.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(features)

            # Your confidence/forgetting aware loss — pass tracker weights as before
            loss = criterion(outputs, labels, your_package.get_weights())
            loss.backward()
            optimizer.step()

        # End of partition: dataset goes out of scope → RAM freed
        # Prefetch thread is already loading the next partition in background
```

---

### Connecting to Apache Ozone

```python
import pyarrow.fs as pafs

fs = pafs.S3FileSystem(
    access_key="your-access-key",
    secret_key="your-secret-key",
    endpoint_override="http://ozone-s3g-host:9878"   # Ozone S3 Gateway address
)

# Option A: hardcoded list
partition_paths = [
    "bucket-name/dataset/part-0000.parquet",
    "bucket-name/dataset/part-0001.parquet",
    "bucket-name/dataset/part-0002.parquet",
]

# Option B: discover dynamically from Ozone directory
file_infos = fs.get_file_info(pafs.FileSelector("bucket-name/dataset/", recursive=False))
partition_paths = sorted([f.path for f in file_infos if f.path.endswith(".parquet")])
```

---

### Entry point

```python
import torch

model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = YourConfidenceForgettingLoss()
your_package = YourExistingPackage()

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    your_package=your_package,
    fs=fs,
    partition_paths=partition_paths,
    feature_cols=["feat_1", "feat_2", "feat_3"],   # replace with actual column names
    label_col="target"                              # replace with actual label column
)
```

---

## What to Plug In

The following placeholders must be replaced with calls matching your existing package's API:

| Placeholder | Replace with |
|---|---|
| `your_package.create_dataloader(dataset)` | Your existing DataLoader creation method |
| `your_package.update_trackers(partition_idx, dataset)` | Your confidence/forgetting tracker update call |
| `your_package.get_weights()` | However your loss reads per-sample tracker weights |
| `criterion(outputs, labels, weights)` | Your custom loss function signature |
| `YourModel()` | Your neural network class |
| `YourExistingPackage()` | Your package class instantiation |

---

## Key Design Properties

**Tracker state persists across partitions**
The `your_package` object lives outside the partition loop. Confidence and forgetting history accumulates naturally across all partitions — no change needed to your tracker logic.

**Existing package is completely untouched**
`PartitionDataset` is a standard `torch.utils.data.Dataset`. Whatever your package does with a Dataset today works identically here. Only the *source* of the dataset changes, not how it is consumed.

**Prefetch queue size = 1 (safe default)**
At most 2 partitions in RAM: the active one and the one being loaded. If your Ozone network is slow and you want more buffer, increase `queue.Queue(maxsize=2)` — but this keeps 3 partitions in RAM simultaneously.

**Backpressure is automatic**
`queue.Queue(maxsize=1)` blocks the background thread if training hasn't finished the current partition. The prefetch thread never runs ahead unconditionally.

**No changes to DataLoader**
Standard `DataLoader` is used per-partition inside your existing package. `num_workers`, `batch_size`, `pin_memory`, `persistent_workers` can all be set as usual inside `create_dataloader`.
