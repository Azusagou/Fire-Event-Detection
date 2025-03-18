import os
import sys
import numpy as np
import torch

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FireEventDataset
import models
import baseline

if __name__ == '__main__':
    # 加载数据
    dataset_name = "spruce_oak_pmma_pur_chipboard"  # 数据集名称
    sample_rate = 32000  # 采样率
    hdf5_path = "dataset_{}_sr_{}.hdf5".format(dataset_name, sample_rate)
    augment = False  # 是否使用数据增强

    # 加载验证集和测试集
    valid_dataset = FireEventDataset(hdf5_path, indice_key='valid_indices', augment=augment)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    test_dataset = FireEventDataset(hdf5_path, indice_key='test_indices', augment=augment)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 定义损失函数
    loss_function = torch.nn.BCELoss()

    # 加载模型
    window_size = 1024  # 窗口大小
    hop_size = 320      # 步长
    mel_bins = 64       # 梅尔频率滤波器组数量
    fmin = 50          # 最小频率
    fmax = 14000       # 最大频率
    classes_num = 1     # 类别数量
    device = torch.device(sys.argv[1])  # 计算设备

    # 加载预训练模型
    experiment_path = 'experiments/baseline'
    model = models.Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
            fmin=fmin, fmax=fmax, classes_num=classes_num)
    model = model.to(device=device)
    model.load_state_dict(torch.load(os.path.join(experiment_path, "best_model.ckpt"), map_location=device))
    model.eval()

    # 评估模型性能
    valid_loss, valid_acc, ys_true_valid, ys_pred_valid = baseline.evaluate(model, valid_loader, loss_function, device)
    test_loss, test_acc, ys_true_test, ys_pred_test = baseline.evaluate(model, test_loader, loss_function, device)

    ##############################################################################
    # 绘制图6：ROC曲线
    ##############################################################################
    fpr, tpr, thresholds = metrics.roc_curve(ys_true_valid, ys_pred_valid)
    threshold_idx = int(np.sum(fpr == 0.0))-1

    # 设置绘图参数
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    scale = 3.54330709/4

    # 绘制ROC曲线
    fig = plt.figure(figsize=(scale*4, scale*3))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(fpr[threshold_idx:], tpr[threshold_idx:], color='k', ls='solid')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    plt.savefig("figure_6.pdf", bbox_inches='tight')
    adjusted_thr = thresholds[threshold_idx]

    ##############################################################################
    # 绘制图5：混淆矩阵
    ##############################################################################

    # 使用默认阈值0.5的混淆矩阵
    thr = 0.5
    true = ys_true_test == 1
    pred = ys_pred_test > thr

    cm = confusion_matrix(true, pred, normalize='true')

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    scale = 3.54330709/4

    # 绘制第一个混淆矩阵
    fig = plt.figure(figsize=(scale*4, scale*3))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, ax=ax, annot=True, cmap='Greys')
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")
    ax.set_title('Confusion Matrix')
    plt.savefig("figure_5_a.pdf", bbox_inches='tight')

    # 计算评估指标
    accuracy_1 = np.mean(true == pred)
    f1_1 = f1_score(true, pred)
    precision_1 = precision_score(true, pred)
    recall_1 = recall_score(true, pred)

    # 使用调整后的阈值的混淆矩阵
    thr = adjusted_thr
    true = ys_true_test == 1
    pred = ys_pred_test > thr

    cm = confusion_matrix(true, pred, normalize='true')

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    scale = 3.54330709/4

    # 绘制第二个混淆矩阵
    fig = plt.figure(figsize=(scale*4, scale*3))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, ax=ax, annot=True, cmap='Greys')
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")
    ax.set_title('Confusion Matrix')
    plt.savefig("figure_5_b.pdf", bbox_inches='tight')

    # 计算评估指标
    accuracy_2 = np.mean(true == pred)
    f1_2 = f1_score(true, pred)
    precision_2 = precision_score(true, pred)
    recall_2 = recall_score(true, pred)

    ##############################################################################
    # 生成表5：不同阈值下的评估指标
    ##############################################################################

    print("--------------------------------")
    print("Metric      thr={:.2f}    thr={:.2f}".format(0.5, adjusted_thr))
    print("--------------------------------")
    print("Accuracy    {:.2f}%       {:.2f}%".format(accuracy_1*100, accuracy_2*100))
    print("Precision   {:.2f}%       {:.2f}%".format(precision_1*100, precision_2*100))
    print("Recall      {:.2f}%       {:.2f}%".format(recall_1*100, recall_2*100))
    print("F-score     {:.2f}%       {:.2f}%".format(f1_1*100, f1_2*100))
    print("--------------------------------")      
      