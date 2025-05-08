# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY requirements.txt .

# 安装Python依赖
RUN pip install -r requirements.txt

# 只复制运行所需的文件
COPY fire_event_detection_app.py .
COPY bocd_detector.py .
COPY model.onnx .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 创建必要的目录
RUN mkdir -p detection_results && \
    chown -R nobody:nogroup /app

# 切换到非root用户
USER nobody

# 暴露端口（用于Gradio界面）
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# 启动应用
CMD ["python", "fire_event_detection_app.py"]
