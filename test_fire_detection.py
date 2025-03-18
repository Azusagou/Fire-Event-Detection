import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import onnxruntime as ort
from prepare_data import split_into_segments
from bocd_detector import BayesianOnlineChangePointDetection

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 模型参数设置
sample_rate = 32000  # 采样率
segment_time = 5     # 音频片段时长(秒)

def load_onnx_model(model_path="model.onnx"):
    """加载ONNX模型"""
    print("正在加载ONNX模型...")
    start_time = time.time()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX模型文件不存在: {model_path}")
    
    # 设置ONNX运行时选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # 设置内部线程数
    
    # 创建推理会话
    providers = ['CPUExecutionProvider']  # 如果有GPU，可以添加'CUDAExecutionProvider'
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    
    elapsed_time = time.time() - start_time
    print(f"ONNX模型加载完成，耗时: {elapsed_time:.2f}秒")
    
    return session

def process_audio_file(audio_path, session, bocd_detector):
    """处理音频文件
    
    Args:
        audio_path: 音频文件路径
        session: ONNX会话
        bocd_detector: BOCD检测器
        
    Returns:
        dict: 处理结果
    """
    print(f"开始处理音频: {audio_path}")
    
    # 加载音频文件
    print("正在加载音频...")
    load_start = time.time()
    wave, _ = librosa.load(audio_path, sr=sample_rate, mono=True, res_type='kaiser_fast')
    print(f"音频加载完成，耗时: {time.time() - load_start:.2f}秒")
    print(f"音频长度: {len(wave)/sample_rate:.2f}秒")
    
    # 分割音频为固定长度片段
    print("正在分割音频...")
    split_start = time.time()
    segments = split_into_segments(wave, sample_rate=sample_rate, segment_time=segment_time)
    print(f"音频分割完成，耗时: {time.time() - split_start:.2f}秒")
    print(f"分割为 {len(segments)} 个片段")
    
    if len(segments) == 0:
        raise ValueError("音频分割失败，无法进行预测")
    
    # 预测每个片段
    print("正在进行预测...")
    pred_start = time.time()
    probabilities = []
    
    for i, segment in enumerate(segments):
        # 准备输入数据
        input_data = segment.reshape(1, -1).astype(np.float32)
        
        # ONNX模型推理
        ort_inputs = {session.get_inputs()[0].name: input_data}
        ort_outputs = session.run(None, ort_inputs)
        prob = ort_outputs[0][0][0]  # 获取预测概率
        
        probabilities.append(prob)
        print(f"片段 {i+1}/{len(segments)} 预测完成: 概率 {prob:.2f}")
    
    print(f"预测完成，耗时: {time.time() - pred_start:.2f}秒")
    
    # 应用BOCD进行变点检测
    print("正在进行变点检测...")
    bocd_start = time.time()
    detection_results = bocd_detector.detect(np.array(probabilities))
    print(f"变点检测完成，耗时: {time.time() - bocd_start:.2f}秒")
    
    # 返回结果
    return {
        'audio_path': audio_path,
        'audio_duration': len(wave)/sample_rate,
        'num_segments': len(segments),
        'probabilities': probabilities,
        'detection_results': detection_results
    }

def visualize_and_save_results(results, output_dir="test_results"):
    """可视化检测结果并保存
    
    Args:
        results: 处理结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio_path = results['audio_path']
    probabilities = results['probabilities']
    detection_results = results['detection_results']
    
    # 获取音频文件名
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 可视化结果
    bocd_detector = BayesianOnlineChangePointDetection()
    fig = bocd_detector.visualize(
        np.array(probabilities), 
        detection_results, 
        segment_duration=segment_time
    )
    
    # 保存图像
    fig_path = os.path.join(output_dir, f"{base_name}_detection.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    
    # 保存文本结果
    txt_path = os.path.join(output_dir, f"{base_name}_results.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"音频文件: {audio_path}\n")
        f.write(f"音频时长: {results['audio_duration']:.2f} 秒\n")
        f.write(f"分段数量: {results['num_segments']}\n\n")
        
        # 变点信息
        change_points = detection_results['change_points']
        if change_points:
            f.write(f"检测到 {len(change_points)} 个变点:\n")
            for i, cp in enumerate(change_points):
                f.write(f"  变点 {i+1}: {cp * segment_time:.2f} 秒\n")
        else:
            f.write("未检测到明显变点\n")
        
        # 火灾事件区间信息
        event_intervals = detection_results['event_intervals']
        if event_intervals:
            f.write(f"\n检测到 {len(event_intervals)} 个火灾事件:\n")
            for i, (start, end) in enumerate(event_intervals):
                start_time = start * segment_time
                end_time = end * segment_time
                duration = end_time - start_time
                f.write(f"  事件 {i+1}: {start_time:.2f}秒 - {end_time:.2f}秒 (持续 {duration:.2f}秒)\n")
        else:
            f.write("\n未检测到火灾事件\n")
        
        # 统计信息
        avg_prob = np.mean(probabilities)
        smoothed_avg_prob = np.mean(detection_results['filtered_probs'])
        f.write(f"\n原始平均概率: {avg_prob:.2f}\n")
        f.write(f"平滑后平均概率: {smoothed_avg_prob:.2f}\n")
    
    print(f"结果已保存到: {output_dir}")
    print(f"  - 图像: {fig_path}")
    print(f"  - 文本: {txt_path}")

def main():
    """主函数"""
    # 创建BOCD检测器
    bocd_detector = BayesianOnlineChangePointDetection(
        threshold=0.3,       # 变点检测阈值
        smoothing_window=5   # 平滑窗口大小
    )
    
    # 加载ONNX模型
    session = load_onnx_model("model.onnx")
    
    # 测试文件路径
    test_file = "test.mp3"  # 可以替换为其他测试音频
    
    # 处理音频文件
    results = process_audio_file(test_file, session, bocd_detector)
    
    # 可视化和保存结果
    visualize_and_save_results(results)

if __name__ == "__main__":
    main() 