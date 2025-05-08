import os
import time
import numpy as np
import librosa
import gradio as gr
import onnxruntime as ort
import matplotlib.pyplot as plt
import tempfile
import matplotlib as mpl
from bocd_detector import BayesianOnlineChangePointDetection

def split_into_segments(wave, sample_rate, segment_time):
    """ Split a wave into segments of segment_size. Repeat signal to get equal
    length segments.
    将音频波形分割成等长片段
    
    Args:
        wave: 音频波形数据
        sample_rate: 采样率
        segment_time: 每个片段的时长(秒)
        
    Returns:
        list: 分割后的音频片段列表
    """
    segment_size = sample_rate * segment_time
    wave_size = wave.shape[0]

    nb_remove = wave_size % segment_size
    if nb_remove > 0:
        truncated_wave = wave[:-nb_remove]
    else:
        truncated_wave = wave

    if not truncated_wave.shape[0] % segment_size == 0:
       raise ValueError("reapeated wave not even multiple of segment size")

    nb_segments = int(truncated_wave.shape[0]/segment_size)
    segments = np.split(truncated_wave, nb_segments, axis=0)

    return segments

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 模型参数设置
sample_rate = 32000  # 采样率
window_size = 1024   # 窗口大小
hop_size = 320       # 步长
mel_bins = 64        # 梅尔频率滤波器组数量
fmin = 50           # 最小频率
fmax = 14000        # 最大频率
classes_num = 1      # 类别数量
segment_time = 5     # 音频片段时长(秒)

# 全局变量存储ONNX会话和BOCD检测器
global_session = None
bocd_detector = BayesianOnlineChangePointDetection(
    threshold=0.2,  # 变点检测阈值
    smoothing_window=2,  # 平滑窗口大小
    trend_memory=0.7  # 时序趋势记忆系数
)

def load_onnx_model():
    """加载ONNX模型"""
    global global_session
    
    if global_session is not None:
        return global_session
    
    print("正在加载ONNX模型...")
    start_time = time.time()
    
    # 创建ONNX运行时会话
    model_path = "model.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX模型文件不存在: {model_path}")
    
    # 设置ONNX运行时选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # 设置内部线程数
    
    # 创建推理会话
    providers = ['CPUExecutionProvider']  # 如果有GPU，可以添加'CUDAExecutionProvider'
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    
    global_session = session
    
    elapsed_time = time.time() - start_time
    print(f"ONNX模型加载完成，耗时: {elapsed_time:.2f}秒")
    
    return session

def predict_with_bocd(audio_path):
    """使用CNN模型和BOCD检测器对音频进行火灾事件检测
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        tuple: (文本结果, 可视化图表)
    """
    if audio_path is None:
        return "请上传音频文件", None
    
    try:
        start_time = time.time()
        print(f"开始处理音频: {audio_path}")
        
        # 加载ONNX模型
        try:
            session = load_onnx_model()
        except Exception as e:
            return f"模型加载失败: {str(e)}\n请检查ONNX模型文件是否存在。", None
        
        # 加载音频文件
        try:
            print("正在加载音频...")
            load_start = time.time()
            wave, _ = librosa.load(audio_path, sr=sample_rate, mono=True, res_type='kaiser_fast')
            print(f"音频加载完成，耗时: {time.time() - load_start:.2f}秒")
            print(f"音频长度: {len(wave)/sample_rate:.2f}秒")
        except Exception as e:
            return f"音频加载失败: {str(e)}\n请检查音频文件格式是否正确。", None
        
        # 分割音频为固定长度片段
        print("正在分割音频...")
        split_start = time.time()
        segments = split_into_segments(wave, sample_rate=sample_rate, segment_time=segment_time)
        print(f"音频分割完成，耗时: {time.time() - split_start:.2f}秒")
        print(f"分割为 {len(segments)} 个片段")
        
        if len(segments) == 0:
            return "音频分割失败，无法进行预测。", None
        
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
        
        # 对于非常短的序列，特殊处理
        if len(probabilities) < 3:
            print("警告: 音频片段过少，无法进行可靠的变点检测")
            # 创建简单的结果
            detection_results = {
                'filtered_probs': np.array(probabilities),  # 不进行平滑
                'change_points': [],  # 没有变点
                'event_intervals': [(0, len(probabilities)-1)] if np.mean(probabilities) > 0.5 else []  # 如果平均概率>0.5，则视为整个序列是一个事件
            }
        else:
            # 正常进行变点检测
            detection_results = bocd_detector.detect(np.array(probabilities))
            
        print(f"变点检测完成，耗时: {time.time() - bocd_start:.2f}秒")
        
        # 可视化结果
        fig = visualize_detection_results(np.array(probabilities), detection_results, segment_time)
        
        # 生成文本结果
        result = "火灾事件检测结果:\n\n"
        
        # 添加变点信息
        change_points = detection_results['change_points']
        if change_points:
            result += f"检测到 {len(change_points)} 个变点:\n"
            for i, cp in enumerate(change_points):
                result += f"变点 {i+1}: {cp * segment_time:.2f} 秒\n"
        else:
            result += "未检测到明显变点\n"
        
        # 添加火灾事件区间信息
        event_intervals = detection_results['event_intervals']
        if event_intervals:
            result += f"\n检测到 {len(event_intervals)} 个火灾事件:\n"
            for i, (start, end) in enumerate(event_intervals):
                start_time_sec = start * segment_time
                end_time_sec = end * segment_time
                duration = end_time_sec - start_time_sec
                result += f"事件 {i+1}: {start_time_sec:.2f}秒 - {end_time_sec:.2f}秒 (持续 {duration:.2f}秒)\n"
        else:
            result += "\n未检测到火灾事件\n"
        
        # 添加总体统计信息
        avg_prob = np.mean(probabilities)
        smoothed_avg_prob = np.mean(detection_results['filtered_probs'])
        result += f"\n原始平均概率: {avg_prob:.2f}\n"
        result += f"平滑后平均概率: {smoothed_avg_prob:.2f}\n"
        
        # 添加处理耗时信息
        elapsed_time = time.time() - start_time
        result += f"\n总处理耗时: {elapsed_time:.2f}秒"
        
        return result, fig
    
    except Exception as e:
        import traceback
        error_msg = f"处理过程中出错: {str(e)}\n\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg, None

def visualize_detection_results(probabilities, detection_results, segment_duration=5):
    """可视化检测结果
    
    Args:
        probabilities: 原始概率序列
        detection_results: detect()方法返回的检测结果
        segment_duration: 每个片段的持续时间(秒)
        
    Returns:
        matplotlib图表对象
    """
    smoothed_probs = detection_results['filtered_probs']
    change_points = detection_results['change_points']
    event_intervals = detection_results['event_intervals']
    change_point_probs = detection_results.get('change_point_probs', None)
    
    # 确保原始概率和平滑概率长度一致
    if len(probabilities) != len(smoothed_probs):
        print(f"警告: 原始概率长度({len(probabilities)})和平滑概率长度({len(smoothed_probs)})不一致")
        min_len = min(len(probabilities), len(smoothed_probs))
        probabilities = probabilities[:min_len]
        smoothed_probs = smoothed_probs[:min_len]
    
    # 创建时间轴
    time_axis = np.arange(len(probabilities)) * segment_duration
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
    fig.suptitle('火灾事件检测结果', fontsize=14)
    
    # 上半部分显示原始概率和平滑后的概率
    ax1.plot(time_axis, probabilities, 'b-', alpha=0.5, label='原始概率')
    ax1.plot(time_axis, smoothed_probs, 'g-', label='平滑后的概率')
    
    # 在上半部分标记事件区间
    for start, end in event_intervals:
        start = min(start, len(time_axis)-1)
        end = min(end, len(time_axis)-1)
        ax1.axvspan(start*segment_duration, end*segment_duration, 
                   color='r', alpha=0.2)
        mid_point = (start + end) / 2
        ax1.text(mid_point*segment_duration, 0.9, "火灾事件", 
                horizontalalignment='center', color='r', fontsize=10)
    
    ax1.grid(True)
    ax1.set_xlabel('时间 (秒)', fontsize=10)
    ax1.set_ylabel('火灾概率', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    
    # 下半部分显示变点概率
    if change_point_probs is not None:
        threshold = bocd_detector.threshold
        below_threshold = change_point_probs < threshold
        above_threshold = ~below_threshold
        
        # 绘制低于阈值的点（绿色）
        if np.any(below_threshold):
            ax2.plot(time_axis[below_threshold], change_point_probs[below_threshold], 
                    'g-', label=f'变点概率 < {threshold}')
        
        # 绘制高于阈值的点（红色）
        if np.any(above_threshold):
            ax2.plot(time_axis[above_threshold], change_point_probs[above_threshold], 
                    'r-', label=f'变点概率 ≥ {threshold}')
        
        # 绘制阈值线
        ax2.axhline(y=threshold, color='k', linestyle='--', alpha=0.5, 
                   label=f'阈值 ({threshold})')
        
        # 标记检测到的变点
        for cp in change_points:
            if cp < len(time_axis):
                ax2.axvline(x=cp*segment_duration, color='r', linestyle=':', alpha=0.7)
                ax2.text(cp*segment_duration, 1.02, f'变点\n{cp*segment_duration:.1f}s', 
                        horizontalalignment='center', fontsize=8)
    
    ax2.grid(True)
    ax2.set_xlabel('时间 (秒)', fontsize=10)
    ax2.set_ylabel('变点概率', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    return fig

def save_detection_results(audio_path):
    """保存检测结果到文件
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        str: 保存结果的信息
    """
    try:
        if audio_path is None:
            return "请先上传音频文件"
        
        # 获取检测结果
        result_text, fig = predict_with_bocd(audio_path)
        
        # 创建结果目录
        results_dir = "detection_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成文件名
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_base = f"{base_name}_{timestamp}"
        
        # 保存文本结果
        text_path = os.path.join(results_dir, f"{result_base}_result.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        # 保存图表
        if fig:
            fig_path = os.path.join(results_dir, f"{result_base}_visualization.png")
            fig.savefig(fig_path, dpi=300)
        
        return f"检测结果已保存到:\n文本: {text_path}\n图表: {fig_path}"
    
    except Exception as e:
        return f"保存结果失败: {str(e)}"

# 预加载模型
print("应用启动中，预加载ONNX模型...")
try:
    load_onnx_model()
    print("ONNX模型预加载完成")
except Exception as e:
    print(f"ONNX模型预加载失败: {str(e)}")

# 创建Gradio界面
with gr.Blocks(title="火灾事件音频检测系统 (增强版)") as demo:
    gr.Markdown("# 火灾事件音频检测系统 (增强版)")
    gr.Markdown("上传音频文件，系统将分析是否包含火灾事件声音，并使用贝叶斯在线变点检测(BOCD)技术识别火灾事件的开始和结束。")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="上传音频文件")
            
            with gr.Row():
                detect_btn = gr.Button("进行检测", variant="primary")
                save_btn = gr.Button("保存结果")
        
        with gr.Column(scale=2):
            result_text = gr.Textbox(label="检测结果", lines=12)
            result_plot = gr.Plot(label="检测可视化")
    
    detect_btn.click(predict_with_bocd, inputs=audio_input, outputs=[result_text, result_plot])
    save_btn.click(save_detection_results, inputs=audio_input, outputs=result_text)
    
    gr.Markdown("## 说明")
    gr.Markdown("""
    - 本系统将音频分割成5秒的片段进行分析
    - 使用CNN14深度学习模型提取音频特征并预测每个片段的火灾概率
    - 贝叶斯在线变点检测(BOCD)算法分析概率序列的变化，识别火灾事件的开始和结束
    - 可视化结果展示原始概率、平滑后的概率，以及检测到的火灾事件区间
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # 配置为Docker环境 