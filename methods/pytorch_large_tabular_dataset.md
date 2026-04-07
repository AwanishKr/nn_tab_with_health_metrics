# PyTorch DataLoader for Large Tabular Data on HDFS / Ozone

## Problem

The dataset is **tabular and numerical**, too large to fit entirely in RAM. It is stored on a distributed filesystem — either **HDFS** or **Apache Ozone** (via its S3-compatible gateway). We need a PyTorch `Dataset` + `DataLoader` setup that:

- Loads data **lazily from disk**, one batch at a time
- Works natively with HDFS and Ozone
- Is efficient enough for training workloads

---

## Chosen Approach: Parquet + PyArrow

### Why Parquet + PyArrow?

| Property | Benefit |
|---|---|
| Columnar format | Read only the columns you need |
| Row group metadata | Build a full row index by reading metadata only — no data loaded upfront |
| PyArrow HDFS connector | Native `HadoopFileSystem` support |
| PyArrow S3 connector | Works with Ozone's S3-compatible gateway |
| Fast numerical I/O | Much faster than CSV for float/int data |

### Dataset Layout on HDFS / Ozone

Split the dataset into multiple Parquet files (row group partitions), each small enough to be cached comfortably in a single worker process:

```
HDFS/Ozone
└── data/
    ├── part-0000.parquet   # ~256MB each
    ├── part-0001.parquet
    ├── part-0002.parquet
    └── ...
```

---

## Full Implementation

### Custom Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import numpy as np


class HDFSOzoneTabularDataset(Dataset):
    def __init__(self, fs, file_paths: list[str], label_col: str, feature_cols: list[str]):
        """
        Args:
            fs           : PyArrow filesystem — HadoopFileSystem (HDFS) or S3FileSystem (Ozone)
            file_paths   : List of parquet file paths on the remote filesystem
            label_col    : Name of the target/label column
            feature_cols : List of feature column names
        """
        self.fs = fs
        self.file_paths = file_paths
        self.label_col = label_col
        self.feature_cols = feature_cols

        # Build row index from parquet metadata only — does NOT load data into RAM
        self.index = []
        for file_idx, path in enumerate(file_paths):
            meta = pq.read_metadata(path, filesystem=fs)
            num_rows = meta.num_rows
            self.index.extend((file_idx, row) for row in range(num_rows))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, row_idx = self.index[idx]

        # Cache the current parquet file in the worker process memory.
        # Each DataLoader worker is a separate process, so caches don't conflict.
        if not hasattr(self, '_cache_file_idx') or self._cache_file_idx != file_idx:
            table = pq.read_table(
                self.file_paths[file_idx],
                filesystem=self.fs,
                columns=self.feature_cols + [self.label_col]
            )
            self._cached_table = table.to_pandas()
            self._cache_file_idx = file_idx

        row = self._cached_table.iloc[row_idx]

        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        return features, label
```

---

### Connecting to HDFS

```python
fs = pafs.HadoopFileSystem(
    host="namenode-host",   # HDFS NameNode hostname or IP
    port=8020,
    user="your-user"
)
```

### Connecting to Ozone (S3-compatible gateway)

```python
fs = pafs.S3FileSystem(
    access_key="your-access-key",
    secret_key="your-secret-key",
    endpoint_override="http://ozone-s3g-host:9878"  # Ozone S3 Gateway endpoint
)
```

---

### Wiring up the DataLoader

```python
file_paths = [
    "data/part-0000.parquet",
    "data/part-0001.parquet",
    "data/part-0002.parquet",
    # add all partition files here
]

feature_cols = ["feat_1", "feat_2", "feat_3"]  # replace with your actual column names
label_col = "target"                            # replace with your actual label column

dataset = HDFSOzoneTabularDataset(
    fs=fs,
    file_paths=file_paths,
    label_col=label_col,
    feature_cols=feature_cols
)

loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    num_workers=8,              # parallel workers — each opens its own FS connection
    pin_memory=True,            # faster CPU → GPU transfer (use when training on CUDA)
    persistent_workers=True     # keep worker processes alive between epochs
)
```

---

### Training Loop

```python
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()  # or CrossEntropyLoss for classification

for epoch in range(num_epochs):
    for features, labels in loader:
        features = features.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Key Design Decisions

### 1. Index built from metadata only
`pq.read_metadata()` reads only the Parquet file footer (row counts, schema, statistics) — it does not load any row data. The full row index `(file_idx, row_idx)` is built cheaply at startup and lives in RAM as a list of integer tuples.

### 2. Per-worker file caching in `__getitem__`
Instead of re-reading the Parquet file from HDFS/Ozone on every single row access, the worker caches the current file as a Pandas DataFrame in its own process memory. When the DataLoader moves to a new file partition, the cache is refreshed. Since each `num_workers` process is isolated, there is no lock contention.

### 3. `num_workers` for parallel I/O
Each worker opens its own independent connection to HDFS/Ozone and prefetches batches while the GPU is busy with the previous batch. This hides network/disk latency effectively.

### 4. `pin_memory=True`
Allocates batch tensors in pinned (page-locked) CPU memory, enabling faster async transfer to the GPU via DMA. Recommended whenever training on CUDA.

### 5. `persistent_workers=True`
Prevents worker processes from being killed and re-spawned between epochs, which saves the overhead of re-establishing HDFS/Ozone connections each epoch.

---

## Dependencies

```bash
pip install torch pyarrow pandas
```

For HDFS specifically, PyArrow requires `libhdfs` which ships with Hadoop. Ensure `HADOOP_HOME` and `JAVA_HOME` are set in your environment:

```bash
export HADOOP_HOME=/path/to/hadoop
export JAVA_HOME=/path/to/jdk
export CLASSPATH=$(hadoop classpath --glob)
```

For Ozone (S3 gateway), no extra native libraries are needed beyond PyArrow.

---

## Alternatives Considered

| Option | Verdict | Reason |
|---|---|---|
| **Parquet + PyArrow** | ✅ Recommended | Native HDFS/S3, columnar, metadata index |
| HDF5 + h5py | ⚠️ Okay for HDFS only | No native Ozone support |
| CSV + pandas | ❌ Avoid | No random row access, slow, large file sizes |
| Petastorm (Uber) | ⚠️ Heavy | Also Parquet-based but adds significant complexity |
