import os
import sys
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """卷积块模块
    
    包含两个卷积层和批归一化层的基本卷积块
    """
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        
        # 第二个卷积层                      
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        """初始化模块的权重"""
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """前向传播
        
        Args:
            input: 输入张量
            pool_size: 池化窗口大小
            pool_type: 池化类型('max', 'avg', 或 'avg+max')
            
        Returns:
            池化后的特征图
        """
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14(nn.Module):
    """CNN14模型架构
    
    基于PANNs (Pretrained Audio Neural Networks)的CNN14模型变体，
    用于音频事件检测
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # 频谱图提取器
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # 对数梅尔特征提取器
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # 频谱增强器
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        # 卷积层块
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # 全连接层
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_fireset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        """初始化模型权重"""
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_fireset)
 
    def forward(self, input, mixup_lambda=None):
        """前向传播
        
        Args:
            input: (batch_size, data_length) 输入音频波形
            mixup_lambda: mixup参数(未使用)
            
        Returns:
            模型输出的类别预测概率
        """
        # 提取频谱图特征
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        #if self.training:
        #    x = self.spec_augmenter(x)

        # Mixup on spectrogram
        #if self.training and mixup_lambda is not None:
        #    x = do_mixup(x, mixup_lambda)
        
        # 通过卷积层块
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 时间维度池化
        x = torch.mean(x, dim=3)
        
        # 全局池化
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # 全连接层
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_fireset(x))
        
        return clipwise_output

        #output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        #return output_dict

def create_folder(fd):
    """创建文件夹"""
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def load_checkpoint(model, checkpoint_path=None, device=torch.device('cpu')):
    """加载预训练模型检查点
    
    Args:
        model: 待加载的模型
        checkpoint_path: 检查点文件路径
        device: 计算设备
    """
    if not checkpoint_path:
        checkpoint_path='./panns_data/Cnn14_mAP=0.431.pth'
    print('Checkpoint path: {}'.format(checkpoint_path))

    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 3e8:
        create_folder(os.path.dirname(checkpoint_path))
        zenodo_path = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
        os.system('wget -O "{}" "{}"'.format(checkpoint_path, zenodo_path))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

