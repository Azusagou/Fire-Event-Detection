import torch
import h5py
import numpy as np

class FireEventDataset(torch.utils.data.Dataset):
    """火灾事件音频数据集
    
    用于加载和处理火灾事件的音频数据，支持数据增强
    """
    def __init__(self, hdf5_path, indice_key, augment=False):
        """初始化数据集
        
        Args:
            hdf5_path: HDF5数据文件路径
            indice_key: 数据集划分索引的键名('train_indices', 'valid_indices', 或 'test_indices')
            augment: 是否使用数据增强
        """
        with h5py.File(hdf5_path, 'r') as f:
            # 加载波形数据和标签
            dset_x = f['wave_segments'][:,:]
            dset_y = f['class_labels'][:,:]
            dset_stats = f['statistics']
            split_indices = f[indice_key]
            self.x = dset_x[split_indices]
            self.y = dset_y[split_indices]
            self.mean, self.variance, self.sample_rate = dset_stats[0]
            
            # 计算类别平衡
            class_0 = np.sum(self.y == 0) / self.y.shape[0]
            class_1 = np.sum(self.y == 1) / self.y.shape[0]
            print(indice_key, " shape: {}, balance: (0 : {}, 1 : {})".format(self.x.shape, class_0, class_1))
        
        self.augment = augment
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.x)
    
    def __getitem__(self, idx):
        """获取数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (音频波形, 标签)
        """
        x = self.x[idx]
        y = self.y[idx]
        
        # 如果启用数据增强，使用简单的数据增强方法
        if self.augment and np.random.random() < 0.5:
            # 添加随机噪声
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.randn(*x.shape) * noise_level
            x = x + noise
            
            # 随机时间偏移
            if np.random.random() < 0.5:
                shift_amount = int(np.random.uniform(-0.1, 0.1) * len(x))
                x = np.roll(x, shift_amount)
        
        return x, y

