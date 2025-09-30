"""
Data loading utilities for phase field datasets.

This module provides PyTorch Dataset and DataLoader implementations for
loading and processing phase field simulation data stored in HDF5 format.
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.multiprocessing import get_context


class PhaseFieldDataset(Dataset):
    """
    Random access dataset for phase field data.

    Suitable for smaller scale datasets where random access is desired.
    Loads data from HDF5 files containing phase field evolution data.

    Args:
        folder_path (str): Path to folder containing HDF5 files.
        ndt (int): Time step interval between consecutive samples. Default: 1.
    """

    def __init__(self, folder_path: str, ndt: int = 1):
        self.folder_path = folder_path
        self.data_info = []
        self.ndt = ndt

        # Process file information
        for file_path in Path(folder_path).glob("*.h5"):
            with h5py.File(file_path, 'r') as f:
                n_samples = f['phi_data'].shape[0] - self.ndt
                # Store file path and sample information
                for i in range(n_samples):
                    self.data_info.append({
                        'file': str(file_path),
                        'index': i
                    })

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (current_phi, next_phi, temp_data)
                - current_phi: Phase field at current time step
                - next_phi: Phase field at next time step
                - temp_data: Temperature field data
        """
        info = self.data_info[idx]
        file_path = info['file']
        phi_idx = info['index']

        # Load data from HDF5 file
        with h5py.File(file_path, 'r') as f:
            current_phi = f['phi_data'][phi_idx].astype(np.float32)
            next_phi = f['phi_data'][phi_idx + self.ndt].astype(np.float32)

            # Handle different temperature data naming conventions
            try:
                temp_data = f['temp_data'][:].astype(np.float32)
            except KeyError:
                temp_data = f['tem_data'][phi_idx].astype(np.float32)

        return (
            torch.from_numpy(current_phi),
            torch.from_numpy(next_phi),
            torch.from_numpy(temp_data)
        )


class PhaseFieldIterableDataset(IterableDataset):
    """
    Iterable dataset for large-scale phase field data.

    Provides memory-efficient streaming of data from HDF5 files,
    suitable for large datasets that don't fit in memory.

    Args:
        folder_path (str): Path to folder containing HDF5 files.
        buffer_size (int): Buffer size for data loading. Default: 1000.
    """

    def __init__(self, folder_path: str, buffer_size: int = 1000):
        super().__init__()
        self.folder_path = folder_path
        self.buffer_size = buffer_size
        self.files = [str(p) for p in Path(folder_path).glob("*.h5")]

        # Pre-compute total number of samples
        self._length = 0
        for file in self.files:
            with h5py.File(file, 'r') as f:
                self._length += f['phi_data'].shape[0] - 1

    def __iter__(self):
        """Iterate through the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        files_per_worker = self.files

        if worker_info is not None:
            # Distribute files among workers
            per_worker = int(np.ceil(len(self.files) / worker_info.num_workers))
            worker_id = worker_info.id
            files_per_worker = self.files[worker_id * per_worker:(worker_id + 1) * per_worker]

        for file_path in files_per_worker:
            with h5py.File(file_path, 'r') as f:
                phi_data = f['phi_data']
                temp_data = f['temp_data'][:]
                total_samples = phi_data.shape[0] - 1

                for start_idx in range(0, total_samples, self.buffer_size):
                    end_idx = min(start_idx + self.buffer_size, total_samples)
                    current_batch = phi_data[start_idx:end_idx]
                    next_batch = phi_data[start_idx + 1:end_idx + 1]

                    for i in range(end_idx - start_idx):
                        yield (
                            torch.from_numpy(current_batch[i].astype(np.float32)),
                            torch.from_numpy(next_batch[i].astype(np.float32)),
                            torch.from_numpy(temp_data.astype(np.float32))
                        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self._length



def create_data_loader(
        folder_path: str,
        batch_size: int,
        ndt: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        use_iterable: bool = False
) -> DataLoader:
    """
    Create a data loader for phase field datasets.

    Args:
        folder_path (str): Path to data folder.
        batch_size (int): Batch size for data loading.
        ndt (int): Time step interval. Default: 1.
        shuffle (bool): Whether to shuffle data. Default: True.
        num_workers (int): Number of worker processes. Default: 4.
        use_iterable (bool): Whether to use iterable dataset. Default: False.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if use_iterable:
        dataset = PhaseFieldIterableDataset(folder_path)
        shuffle = False  # IterableDataset doesn't support shuffle
    else:
        dataset = PhaseFieldDataset(folder_path, ndt=ndt)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if num_workers > 0 else 0,
        multiprocessing_context=get_context('spawn') if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

