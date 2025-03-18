import os
import torch
import models

# 模型参数设置
sample_rate = 32000  # 采样率
window_size = 1024   # 窗口大小
hop_size = 320       # 步长
mel_bins = 64        # 梅尔频率滤波器组数量
fmin = 50           # 最小频率
fmax = 14000        # 最大频率
classes_num = 1      # 类别数量

def convert_to_onnx():
    """将PyTorch模型转换为ONNX格式"""
    print("开始转换模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载PyTorch模型
    print("加载PyTorch模型...")
    model = models.Cnn14(sample_rate=sample_rate, window_size=window_size, 
                        hop_size=hop_size, mel_bins=mel_bins,
                        fmin=fmin, fmax=fmax, classes_num=classes_num)
    
    checkpoint_path = os.path.join('experiments/baseline', "best_model.ckpt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, sample_rate * 5)  # 5秒音频的示例输入
    
    # 设置输出路径
    onnx_path = "model.onnx"
    
    # 导出ONNX模型
    print(f"导出ONNX模型到 {onnx_path}...")
    torch.onnx.export(model,               # 要转换的模型
                     dummy_input,          # 模型输入示例
                     onnx_path,            # 保存路径
                     export_params=True,    # 存储训练好的参数权重
                     opset_version=11,      # ONNX算子集版本
                     do_constant_folding=True,  # 是否执行常量折叠优化
                     input_names=['input'],     # 输入名
                     output_names=['output'],   # 输出名
                     dynamic_axes={'input': {0: 'batch_size'},    # 动态轴
                                 'output': {0: 'batch_size'}})
    
    print("模型转换完成！")
    
    # 验证导出的模型
    import onnx
    print("验证ONNX模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过！")
    
    # 输出模型大小信息
    pytorch_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nPyTorch模型大小: {pytorch_size:.2f} MB")
    print(f"ONNX模型大小: {onnx_size:.2f} MB")

if __name__ == "__main__":
    convert_to_onnx() 