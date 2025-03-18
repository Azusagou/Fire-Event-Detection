#!/usr/bin/env python
# coding: utf-8

import sys
import os
import tqdm
import glob
import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from dataset import FireEventDataset

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import models
from models import load_checkpoint
np.random.seed(42)

def train(model, optimizer, loss_function, train_loader, device):
    """训练模型的一个epoch
    
    Args:
        model: 待训练的模型
        optimizer: 优化器
        loss_function: 损失函数
        train_loader: 训练数据加载器
        device: 计算设备(CPU/GPU)
    
    Returns:
        float: 平均训练损失
    """
    model.train()
    
    running_loss = 0
    count = 0
    for (x, y) in tqdm.tqdm(train_loader):
        x = x.to(device=device)
        y = y.type(torch.FloatTensor).to(device=device)
        
        optimizer.zero_grad()
        
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        
        optimizer.step()
    
        running_loss += loss.item()
        count += 1
    return running_loss / count

def evaluate(model, loader, loss_function, device):
    """评估模型性能
    
    Args:
        model: 待评估的模型
        loader: 数据加载器
        loss_function: 损失函数
        device: 计算设备(CPU/GPU)
    
    Returns:
        tuple: (平均损失, 平均准确率, 真实标签, 预测概率)
    """
    model.eval()
    
    count = 0
    running_acc = 0
    running_loss = 0
    
    ys = []
    ys_pred = []
    for (x, y) in loader:
        x = x.to(device=device)
        y = y.type(torch.FloatTensor).to(device=device)
        
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()
        
        y_pred = y_pred.detach().cpu().numpy()
        y      = y.detach().cpu().numpy().astype(np.int32)

        running_acc += np.mean(y==np.round(y_pred).astype(np.int32))
        
        count+=1
        
        ys.append(y)
        ys_pred.append(y_pred)
    
    return running_loss / count, running_acc / count, np.concatenate(ys), np.concatenate(ys_pred)

def main():
    # 数据集参数设置
    sample_rate = 32000  # 采样率
    dataset_name = "spruce_oak_pmma_pur_chipboard"  # 数据集名称
    hdf5_path = "dataset_{}_sr_{}.hdf5".format(dataset_name, sample_rate)
       
    # PANNs模型参数设置
    window_size = 1024  # 窗口大小
    hop_size = 320      # 步长
    mel_bins = 64       # 梅尔频率滤波器组数量
    fmin = 50          # 最小频率
    fmax = 14000       # 最大频率
    classes_num = 1     # 类别数量

    # 训练参数设置
    epochs = 5000       # 最大训练轮数
    learning_rate = 1e-4  # 学习率
    pretrained = False    # 是否使用预训练模型
    experiment_dir = 'experiments'  # 实验结果保存目录
    patience = 100        # 早停耐心值
    batch_size = 8       # 批次大小

    # 运行模式: ["train_model", "evaluate_model"]
    mode = sys.argv[1]
    device = torch.device(sys.argv[2])

    experiment_name = 'baseline'
    experiment_path = os.path.join(experiment_dir, experiment_name)
    if mode == "train_model":
        # 训练模式
        writer = SummaryWriter(log_dir=experiment_path)

        # 初始化模型
        model = models.Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
                            fmin=fmin, fmax=fmax, classes_num=classes_num)
        model = model.to(device=device)
        # 创建模型副本用于保存最佳模型
        best_model = models.Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
                fmin=fmin, fmax=fmax, classes_num=classes_num)
        best_model = best_model.to(device=device)

        if pretrained:
            load_checkpoint(model, device=device)

        # 初始化优化器和损失函数
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        loss_function = torch.nn.BCELoss()

        # 加载数据集
        train_dataset = FireEventDataset(hdf5_path, indice_key='train_indices', augment=False)
        valid_dataset = FireEventDataset(hdf5_path, indice_key='valid_indices', augment=False)
        test_dataset = FireEventDataset(hdf5_path, indice_key='test_indices', augment=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        # 训练循环控制变量
        best_valid_loss = np.inf
        best_epoch = 0
        epoch = 0
        not_converged = True
        
        # 训练循环
        while not_converged:
            train_loss = train(model, optimizer, loss_function, train_loader, device)
            valid_loss, valid_acc, _, _ = evaluate(model, valid_loader, loss_function, device)
            print("valid loss: {}, acc: {}".format(valid_loss, valid_acc))
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/valid', valid_loss, epoch)
            writer.add_scalar('acc/valid', valid_acc, epoch)

            epoch += 1

            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_model.load_state_dict(model.state_dict())

            # 收敛判断
            if epoch - best_epoch >= patience or epoch > epochs:
                not_converged = False

        # 保存模型
        torch.save(model.state_dict(), os.path.join(experiment_path, "model_epochs_{}.ckpt".format(epoch)))
        torch.save(best_model.state_dict(), os.path.join(experiment_path, "best_model.ckpt"))
        test_loss, test_acc, _, _ = evaluate(model, test_loader, loss_function, device)
        print("test loss: {}, acc: {}".format(test_loss, test_acc))
    
    elif mode == "evaluate_model":
        # 评估模式
        print("Evaluate models...")
        model = models.Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
            fmin=fmin, fmax=fmax, classes_num=classes_num)
        model = model.to(device=device)
        
        # 加载最佳模型
        best_model_path = os.path.join(experiment_path, "best_model.ckpt")
        print("model: ", experiment_path)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        # 加载验证集和测试集
        valid_dataset = FireEventDataset(hdf5_path, indice_key='valid_indices', augment=False)
        valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        test_dataset = FireEventDataset(hdf5_path, indice_key='test_indices', augment=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        loss_function = torch.nn.BCELoss()

        # 评估模型性能
        valid_loss, valid_acc, ys, ys_pred = evaluate(model, valid_loader, loss_function, device)
        test_loss, test_acc, ys, ys_pred = evaluate(model, test_loader, loss_function, device)

        print("Learning rate: {}, accuracy: {} (valid)".format(learning_rate, valid_acc))
        print("Learning rate: {}, accuracy: {} (test)".format(learning_rate, test_acc))

        # 将连续的预测概率转换为二进制类别
        ys = ys == 1
        ys_pred = ys_pred > 0.5

        # 绘制混淆矩阵
        cm = confusion_matrix(ys, ys_pred)
        sns.heatmap(cm, annot=True)
        plt.title("Accuracy: {:.2f}, model: {}".format(test_acc, "CNN14"))
        plt.ylabel("True class")
        plt.xlabel("Predicted class")
        plt.savefig(os.path.join(experiment_path, "confusion_matrix.pdf"))

if __name__ == '__main__':
    main()
