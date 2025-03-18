import os
import time
import numpy as np
import librosa
import gradio as gr
import onnxruntime as ort
from prepare_data import split_into_segments

# 模型参数设置
sample_rate = 32000  # 采样率
window_size = 1024   # 窗口大小
hop_size = 320       # 步长
mel_bins = 64        # 梅尔频率滤波器组数量
fmin = 50           # 最小频率
fmax = 14000        # 最大频率
classes_num = 1      # 类别数量
segment_time = 5     # 音频片段时长(秒)

# 全局变量存储ONNX会话
global_session = None

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

def predict(audio_path):
    """使用ONNX模型预测音频中是否包含火灾事件
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        预测结果
    """
    if audio_path is None:
        return "请上传音频文件"
    
    try:
        start_time = time.time()
        print(f"开始处理音频: {audio_path}")
        
        # 加载ONNX模型
        try:
            session = load_onnx_model()
        except Exception as e:
            return f"模型加载失败: {str(e)}\n请检查ONNX模型文件是否存在。"
        
        # 加载音频文件
        try:
            print("正在加载音频...")
            load_start = time.time()
            wave, _ = librosa.load(audio_path, sr=sample_rate, mono=True, res_type='kaiser_fast')
            print(f"音频加载完成，耗时: {time.time() - load_start:.2f}秒")
            print(f"音频长度: {len(wave)/sample_rate:.2f}秒")
        except Exception as e:
            return f"音频加载失败: {str(e)}\n请检查音频文件格式是否正确。"
        
        # 分割音频为固定长度片段
        print("正在分割音频...")
        split_start = time.time()
        segments = split_into_segments(wave, sample_rate=sample_rate, segment_time=segment_time)
        print(f"音频分割完成，耗时: {time.time() - split_start:.2f}秒")
        print(f"分割为 {len(segments)} 个片段")
        
        if len(segments) == 0:
            return "音频分割失败，无法进行预测。"
        
        # 预测每个片段
        print("正在进行预测...")
        pred_start = time.time()
        predictions = []
        probabilities = []
        
        for i, segment in enumerate(segments):
            # 准备输入数据
            input_data = segment.reshape(1, -1).astype(np.float32)
            
            # ONNX模型推理
            ort_inputs = {session.get_inputs()[0].name: input_data}
            ort_outputs = session.run(None, ort_inputs)
            prob = ort_outputs[0][0][0]  # 获取预测概率
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
            print(f"片段 {i+1}/{len(segments)} 预测完成: 概率 {prob:.4f}")
        
        print(f"预测完成，耗时: {time.time() - pred_start:.2f}秒")
        
        # 汇总结果
        avg_prob = np.mean(probabilities)
        final_pred = 1 if avg_prob > 0.5 else 0
        
        result = f"检测结果: {'检测到火灾事件' if final_pred == 1 else '未检测到火灾事件'}\n"
        result += f"火灾事件概率: {avg_prob:.4f}\n\n"
        
        # 添加每个片段的预测结果
        result += "各片段详细结果:\n"
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result += f"片段 {i+1}: {'火灾事件' if pred == 1 else '非火灾事件'} (概率: {prob:.4f})\n"
        
        elapsed_time = time.time() - start_time
        result += f"\n处理总耗时: {elapsed_time:.2f}秒"
        print(f"预测完成，总耗时: {elapsed_time:.2f}秒")
        
        return result
    
    except Exception as e:
        import traceback
        error_msg = f"处理过程中出错: {str(e)}\n\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg

# 预加载模型
print("应用启动中，预加载ONNX模型...")
try:
    load_onnx_model()
    print("ONNX模型预加载完成")
except Exception as e:
    print(f"ONNX模型预加载失败: {str(e)}")

# 创建Gradio界面
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="上传音频文件"),
    outputs=gr.Textbox(label="检测结果", lines=15),
    title="火灾事件音频检测系统 (baseline)",
    description="上传音频文件，系统将分析是否包含火灾事件声音。系统会将音频分割成5秒的片段进行分析，并给出每个片段的检测结果和整体结果。"
)

# 启动应用
if __name__ == "__main__":
    demo.launch(share=True)  # share=True可以生成一个公共链接，方便分享 