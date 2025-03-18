import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BayesianOnlineChangePointDetection:
    """贝叶斯在线变点检测(BOCD)
    
    用于检测时间序列中的变点，特别适用于火灾事件检测中的概率变化分析
    """
    def __init__(self, hazard_func=None, observation_likelihood=None, 
                threshold=0.5, smoothing_window=3):
        """初始化BOCD检测器
        
        Args:
            hazard_func: 风险函数，用于计算变点发生的先验概率
            observation_likelihood: 观测似然函数，用于计算观测数据的似然
            threshold: 变点检测阈值
            smoothing_window: 平滑窗口大小
        """
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        
        # 如果没有指定风险函数，使用默认的常数风险函数
        if hazard_func is None:
            self.hazard_func = self._constant_hazard
        else:
            self.hazard_func = hazard_func
        
        # 如果没有指定观测似然函数，使用默认的高斯似然函数
        if observation_likelihood is None:
            self.observation_likelihood = self._gaussian_likelihood
        else:
            self.observation_likelihood = observation_likelihood
    
    def _constant_hazard(self, r, hazard_rate=250):
        """常数风险函数
        
        Args:
            r: 运行长度（当前步与上一个变点之间的距离）
            hazard_rate: 风险率，值越小表示变点发生的概率越高
        
        Returns:
            变点发生的先验概率
        """
        return 1.0 / hazard_rate
    
    def _gaussian_likelihood(self, data_point, pred_history, r):
        """高斯似然函数
        
        Args:
            data_point: 当前数据点
            pred_history: 历史预测数据
            r: 运行长度
        
        Returns:
            观测数据的似然
        """
        # 计算历史数据的均值和标准差
        if r < 2:
            # 当运行长度太短时，使用固定参数
            mu = data_point
            sigma = 0.1
        else:
            # 使用历史数据计算参数
            recent_data = pred_history[-r:]
            mu = np.mean(recent_data)
            sigma = np.std(recent_data) + 1e-6  # 防止标准差为0
        
        # 计算似然
        return norm.pdf(data_point, mu, sigma)
    
    def detect(self, probabilities):
        """对概率序列进行变点检测
        
        Args:
            probabilities: 模型预测的概率序列
        
        Returns:
            dict: 包含检测结果的字典：
                - 'filtered_probs': 过滤后的概率序列
                - 'change_points': 检测到的变点位置
                - 'event_intervals': 火灾事件区间 [(start1, end1), (start2, end2), ...]
        """
        # 首先对概率序列进行平滑
        smoothed_probs = self._smooth_probabilities(probabilities)
        
        n = len(smoothed_probs)
        # 运行长度分布，表示当前时刻与上一个变点之间的距离的概率分布
        run_length_dist = np.zeros((n, n))
        # 初始状态：第一个时刻的运行长度为0的概率为1
        run_length_dist[0, 0] = 1
        
        # 检测变点
        change_points = []
        max_run_lengths = []
        
        # 在线变点检测的主循环
        for t in range(1, n):
            # 计算风险函数
            hazard = np.array([self.hazard_func(r) for r in range(t)])
            
            # 计算观测似然
            pred_history = smoothed_probs[:t]
            likelihood = np.array([
                self.observation_likelihood(smoothed_probs[t], pred_history, r) 
                for r in range(t)
            ])
            
            # 计算增长概率
            growth_probs = run_length_dist[t-1, :t] * (1 - hazard) * likelihood
            
            # 计算变点概率
            cp_prob = np.sum(run_length_dist[t-1, :t] * hazard * likelihood)
            
            # 更新运行长度分布
            run_length_dist[t, 1:t+1] = growth_probs
            run_length_dist[t, 0] = cp_prob
            
            # 归一化
            run_length_dist[t, :t+1] /= np.sum(run_length_dist[t, :t+1]) + 1e-9
            
            # 找出当前最可能的运行长度
            max_run_length = np.argmax(run_length_dist[t, :t+1])
            max_run_lengths.append(max_run_length)
            
            # 检测变点
            if run_length_dist[t, 0] > self.threshold:
                change_points.append(t)
        
        # 识别火灾事件区间
        event_intervals = self._identify_fire_events(smoothed_probs, change_points)
        
        return {
            'filtered_probs': smoothed_probs,
            'change_points': change_points,
            'event_intervals': event_intervals
        }
    
    def _smooth_probabilities(self, probabilities):
        """使用滑动窗口平滑概率序列
        
        Args:
            probabilities: 原始概率序列
        
        Returns:
            平滑后的概率序列
        """
        window_size = self.smoothing_window
        smoothed = np.convolve(
            probabilities, 
            np.ones(window_size)/window_size,
            mode='same'
        )
        
        # 处理边界情况
        pad_size = window_size // 2
        for i in range(pad_size):
            # 前边界
            window = probabilities[:i+pad_size+1]
            smoothed[i] = np.mean(window)
            
            # 后边界
            window = probabilities[-(i+pad_size+1):]
            smoothed[-(i+1)] = np.mean(window)
            
        return smoothed
    
    def _identify_fire_events(self, probabilities, change_points):
        """根据变点识别火灾事件区间
        
        Args:
            probabilities: 概率序列
            change_points: 变点位置
        
        Returns:
            list: 火灾事件区间列表 [(start1, end1), (start2, end2), ...]
        """
        if not change_points:
            # 如果没有检测到变点，判断整体概率水平
            if np.mean(probabilities) > 0.5:
                return [(0, len(probabilities)-1)]
            else:
                return []
        
        # 将变点配对，形成区间
        intervals = []
        event_ongoing = False
        start_idx = 0
        
        for cp in change_points:
            # 检查变点前后的概率水平变化
            before_cp = np.mean(probabilities[max(0, cp-self.smoothing_window):cp])
            after_cp = np.mean(probabilities[cp:min(len(probabilities), cp+self.smoothing_window)])
            
            if not event_ongoing and after_cp > 0.5:
                # 火灾开始
                start_idx = cp
                event_ongoing = True
            elif event_ongoing and after_cp < 0.5:
                # 火灾结束
                intervals.append((start_idx, cp))
                event_ongoing = False
        
        # 处理最后一个未闭合的区间
        if event_ongoing:
            intervals.append((start_idx, len(probabilities)-1))
        
        return intervals
    
    def visualize(self, probabilities, detection_results, segment_duration=5):
        """可视化检测结果
        
        Args:
            probabilities: 原始概率序列
            detection_results: detect()方法返回的检测结果
            segment_duration: 每个片段的持续时间(秒)
        """
        smoothed_probs = detection_results['filtered_probs']
        change_points = detection_results['change_points']
        event_intervals = detection_results['event_intervals']
        
        # 创建时间轴
        time_axis = np.arange(len(probabilities)) * segment_duration
        
        plt.figure(figsize=(12, 6))
        
        # 绘制原始概率和平滑后的概率
        plt.plot(time_axis, probabilities, 'b-', alpha=0.5, label='原始概率')
        plt.plot(time_axis, smoothed_probs, 'g-', label='平滑后的概率')
        
        # 绘制变点
        for cp in change_points:
            plt.axvline(x=cp*segment_duration, color='r', linestyle='--', alpha=0.7)
        
        # 高亮火灾事件区间
        for start, end in event_intervals:
            plt.axvspan(start*segment_duration, end*segment_duration, 
                       color='r', alpha=0.2)
            
            # 添加标签
            mid_point = (start + end) / 2
            plt.text(mid_point*segment_duration, 0.9, "火灾事件", 
                    horizontalalignment='center', color='r', fontsize=12)
        
        plt.grid(True)
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('火灾概率', fontsize=12)
        plt.title('火灾事件检测结果', fontsize=14)
        plt.legend(fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        
        return plt.gcf() 