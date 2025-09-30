
import h5py
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader
from pathlib import Path
from torch.multiprocessing import get_context


class PhaseFieldDataset(Dataset):
    """
    随机访问数据集实现，适用于较小规模数据。
    ndt是对于不同数据集需要调
    """

    def __init__(self, folder_path, ndt=1):
        self.folder_path = folder_path
        self.data_info = []
        self.ndt = ndt

        # 预处理文件信息
        for file_path in Path(folder_path).glob("*.h5"):
            with h5py.File(file_path, 'r') as f:
                n_samples = f['phi_data'].shape[0] - self.ndt
                # 存储文件路径和样本信息
                for i in range(n_samples):
                    self.data_info.append({
                        'file': str(file_path),
                        'index': i
                    })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 物理参数
        Q = 1.82
        kev = 8.625e-5
        Tmax = 1073.0

        info = self.data_info[idx]
        file_path = info['file']
        phi_idx = info['index']

        # 每次需要时才打开文件读取数据
        with h5py.File(file_path, 'r') as f:
            current_phi = f['phi_data'][phi_idx].astype(np.float32)
            next_phi = f['phi_data'][phi_idx + self.ndt].astype(np.float32)
            temp_data = f['temp_data'][:].astype(np.float32)

        return (
            torch.from_numpy(current_phi),
            torch.from_numpy(next_phi),
            torch.exp(-Q/ (kev * torch.from_numpy(temp_data))) / np.exp(-Q / (kev * Tmax))
        )


class PhaseFieldIterableDataset(IterableDataset):
    def __init__(self, folder_path, buffer_size=1000):
        super().__init__()
        self.folder_path = folder_path
        self.buffer_size = buffer_size
        self.files = [str(p) for p in Path(folder_path).glob("*.h5")]

        # 预计算总样本数
        self._length = 0
        for file in self.files:
            with h5py.File(file, 'r') as f:
                self._length += f['phi_data'].shape[0] - 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_per_worker = self.files

        if worker_info is not None:
            # 在多进程情况下分配文件
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
                            torch.from_numpy(temp_data.astype(np.float32)) # 放缩
                        )

    def __len__(self):
        return self._length

def create_data_loader(folder_path, batch_size, ndt=1, shuffle=True, num_workers=4, use_iterable=True):
    """
    创建数据加载器，支持两种模式。
    """
    if use_iterable:
        dataset = PhaseFieldIterableDataset(folder_path)
        shuffle = False  # IterableDataset不支持shuffle
    else:
        dataset = PhaseFieldDataset(folder_path, ndt=ndt)

    # 使用 'spawn' 上下文
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if num_workers > 0 else 0,
        multiprocessing_context=get_context('spawn') if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

