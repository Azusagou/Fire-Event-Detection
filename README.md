# 火灾事件音频检测系统 (增强版)

这是一个基于深度学习的火灾事件音频检测系统，通过时序分析和变点检测技术提高检测准确性。系统结合了 CNN14 深度学习模型和贝叶斯在线变点检测(BOCD)，能够分析音频数据中的时序特征，准确识别火灾事件的开始和结束。该系统支持 Docker 容器化部署。

## 系统特点

- **贝叶斯在线变点检测(BOCD)**：实时检测概率序列中的变点，准确识别火灾事件的开始和结束
- **序列处理能力**：将音频分割成固定长度片段，支持任意长度的音频分析
- **可视化工具**：提供检测结果的静态图和详细文本报告，方便分析
- **多种接口**：支持 Web 界面和命令行两种使用方式
- **容器化部署**：支持 Docker 容器化部署，便于系统迁移和扩展
- 支持上传各种格式的音频文件（WAV、MP3 等）
- 提供友好的 Web 界面，方便演示和使用
- 使用 ONNX 优化，提供快速的推理速度

## 项目结构

```
├── wav/                       # 原始音频文件目录
├── experiments/               # 实验结果和模型文件目录
├── dataset.py                 # 数据集处理代码
├── models.py                  # 模型定义代码
├── baseline.py                # 基础模型训练和评估代码
├── bocd_detector.py           # 贝叶斯在线变点检测实现
├── fire_event_detection_app.py # 增强版Web应用（含BOCD检测）
├── test_fire_detection.py     # 命令行测试脚本
├── prepare_data.py            # 数据预处理代码
├── convert_model.py           # 模型格式转换工具
├── Dockerfile                 # Docker构建文件
└── requirements.txt           # 项目依赖文件
```

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA (可选，用于 GPU 加速)
- Docker (可选，用于容器化部署)

### 依赖包

```
torch>=1.7.0
numpy>=1.19.0
librosa>=0.8.0
gradio>=3.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.50.0
h5py>=3.1.0
pandas>=1.1.0
torchlibrosa>=0.0.9
onnxruntime>=1.7.0
scipy>=1.5.0
```

## 部署方式

### 方式一：Docker 部署（推荐）

1. 确保已安装 Docker

```bash
docker --version
```

2. 构建 Docker 镜像

```bash
docker build -t fire-detection-system .
```

3. 运行容器

```bash
docker run -d -p 7860:7860 --name fire-detection fire-detection-system
```

4. 访问系统

```
在浏览器中打开 http://localhost:7860
```

Docker 环境变量配置：

- `MODEL_PATH`: 模型文件路径
- `CUDA_VISIBLE_DEVICES`: GPU 设备选择
- `PORT`: Web 服务端口号

### 方式二：本地部署

1. 克隆或下载本仓库

2. 创建并激活虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 准备数据集

- 将原始音频文件(.WAV)放入`wav`目录
- 确保`spruce_oak_pmma_pur_chipboard.csv`文件存在
- 运行数据预处理：

```bash
python prepare_data.py
```

## 模型训练

### 训练基础模型

```bash
python baseline.py train_model cpu  # 使用CPU训练
# 或
python baseline.py train_model cuda  # 使用GPU训练
```

## 使用说明

### Web 界面

启动增强版的 Web 应用：

```bash
python fire_event_detection_app.py
```

Web 界面提供以下功能：

- 上传音频文件进行分析
- 可视化显示检测结果
- 保存检测结果到文件

### 命令行使用

通过命令行脚本对特定音频文件进行检测：

```bash
python test_fire_detection.py
```

可以修改脚本中的`test_file`变量来指定要分析的音频文件。

### 环境变量配置

在 `docker-compose.yml` 中可以配置以下环境变量：

- `MODEL_PATH`: 模型文件路径
- `CUDA_VISIBLE_DEVICES`: GPU 设备选择
- `PORT`: Web 服务端口号

## 贝叶斯在线变点检测(BOCD)

BOCD 是一种实时检测时间序列中变化点的技术，特别适合火灾检测场景：

1. **增强的概率估计**：综合考虑历史概率和当前概率
2. **趋势分析**：捕捉火灾概率的上升和下降趋势
3. **变点检测**：准确识别火灾开始和结束的时间点
4. **鲁棒性**：减少噪声和短暂波动的影响

BOCD 的工作流程：

1. CNN 模型对每个 5 秒的音频片段预测火灾概率
2. 使用滑动窗口平滑概率序列，减少噪声影响
3. BOCD 算法检测概率序列中的变点
4. 根据变点和概率水平识别火灾事件区间
5. 生成可视化结果和详细报告

## 检测结果说明

系统检测的结果包括：

1. **变点信息**：检测到的概率序列中的变点位置
2. **火灾事件区间**：识别出的火灾开始和结束时间
3. **概率变化图**：显示原始概率和平滑后的概率变化
4. **统计信息**：平均概率等汇总数据

## 参数调优

BOCD 检测器的主要参数：

1. **threshold**（变点检测阈值）：

   - 默认值：0.3
   - 降低阈值会增加变点的敏感度，但可能增加误报
   - 增加阈值会减少误报，但可能导致漏报

2. **smoothing_window**（平滑窗口大小）：
   - 默认值：5
   - 增大窗口会使概率曲线更平滑，减少噪声影响
   - 减小窗口会保留更多细节变化

## 模型说明

### 基础模型(CNN14)

- 基于 PANNs (Pretrained Audio Neural Networks)的 CNN14 结构
- 使用梅尔频谱图作为特征
- 输出：火灾事件概率（0-1 之间）

## 性能优化建议

1. 模型优化

   - 优化 BOCD 参数（变点检测阈值和平滑窗口大小）
   - 考虑使用模型量化减小体积

2. 硬件优化

   - 使用 GPU 加速（推荐）
   - 配置足够的内存（建议 8GB 以上）

3. 使用优化
   - 对于长音频，调整平滑窗口大小可能会影响变点检测的准确性
   - 如果检测结果不理想，尝试调整 BOCD 阈值参数

## 常见问题

1. 音频加载失败

   - 检查音频文件格式是否支持
   - 确保 librosa 库正确安装

2. 模型加载失败

   - 确保 ONNX 模型文件存在
   - 检查 onnxruntime 库是否正确安装

3. 检测结果不准确
   - 调整 BOCD 的阈值参数
   - 调整平滑窗口大小
   - 检查音频质量，可能存在噪声干扰

## 开发计划

- [ ] 添加 Transformer 变体模型
- [ ] 实现模型蒸馏
- [ ] 添加多模态支持
- [ ] 开发实时监控系统
- [ ] 实现移动设备部署
