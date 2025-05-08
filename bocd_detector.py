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
                threshold=0.2, smoothing_window=2, trend_memory=0.7):
        """初始化BOCD检测器
        
        Args:
            hazard_func: 风险函数，用于计算变点发生的先验概率
            observation_likelihood: 观测似然函数，用于计算观测数据的似然
            threshold: 变点检测阈值
            smoothing_window: 平滑窗口大小
            trend_memory: 时序趋势记忆系数（0-1），越大表示越考虑历史趋势
        """
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.trend_memory = trend_memory
        
        # 如果没有指定风险函数，使用默认的常数风险函数
        if hazard_func is None:
            self.hazard_func = self._constant_hazard
        else:
            self.hazard_func = hazard_func
        
        # 如果没有指定观测似然函数，使用默认的概率序列似然函数
        if observation_likelihood is None:
            self.observation_likelihood = self._trend_aware_likelihood
        else:
            self.observation_likelihood = observation_likelihood
    
    def _constant_hazard(self, r, hazard_rate=25):
        """常数风险函数
        
        Args:
            r: 运行长度（当前步与上一个变点之间的距离）
            hazard_rate: 风险率，值越小表示变点发生的概率越高
        
        Returns:
            变点发生的先验概率
        """
        return 1.0 / hazard_rate
    
    def _trend_aware_likelihood(self, data_point, pred_history, r):
        """考虑时序趋势的似然函数
        
        Args:
            data_point: 当前概率值
            pred_history: 历史概率序列
            r: 运行长度
        
        Returns:
            观测数据的似然
        """
        try:
            if r < 3:
                # 当运行长度太短时，使用一个固定似然值
                return 0.1
            
            # 获取最近的历史数据
            recent_data = pred_history[-r:]
            
            # 计算最近历史的均值
            mean_prob = np.mean(recent_data)
            
            # 计算历史趋势（最近n点的斜率）
            trend_window = min(5, r)
            trend_data = recent_data[-trend_window:]
            if len(trend_data) >= 2:
                try:
                    # 使用简单线性回归估计趋势
                    x = np.arange(len(trend_data))
                    slope = np.polyfit(x, trend_data, 1)[0]
                except Exception:
                    # 如果回归计算失败，使用零斜率
                    slope = 0
            else:
                slope = 0
            
            # 根据趋势预测当前点的期望值
            expected_value = mean_prob + slope  # 线性外推
            
            # 根据趋势和均值的混合预测当前值
            predicted_value = self.trend_memory * expected_value + (1 - self.trend_memory) * recent_data[-1]
            predicted_value = max(0, min(1, predicted_value))  # 确保在0-1范围内
            
            # 计算当前数据点与预测值的差异
            diff = abs(data_point - predicted_value)
            
            # 时序状态判断（持续稳定或突变）
            stability = np.std(recent_data)  # 稳定性度量
            
            # 调整似然值计算，考虑稳定性和差异
            if stability < 0.1 and diff > 0.2:  # 从稳定状态突变
                return 0.95  # 非常高的似然值表示很可能是变点
            elif stability < 0.1 and diff < 0.1:  # 继续稳定
                return np.exp(-diff * 10)  # 差异小时似然值高
            elif stability >= 0.1:  # 处于不稳定状态
                # 在不稳定状态下，更加关注是否有明确的突变趋势
                if diff > 0.25:  # 重大变化
                    return 0.9
                else:
                    return np.exp(-diff * 5)
            
            # 兜底返回值，避免返回None
            return 0.5
            
        except Exception as e:
            # 出现异常时，返回一个安全的默认值
            print(f"似然函数计算异常: {str(e)}")
            return 0.5  # 返回中性似然值
    
    def _detect_trends_and_jumps(self, smoothed_probs, window_size=3, jump_threshold=0.15):
        """检测概率序列中的趋势变化和跳变
        
        Args:
            smoothed_probs: 平滑后的概率序列
            window_size: 比较窗口大小
            jump_threshold: 跳变阈值
        
        Returns:
            list: 检测到的变化点索引列表
        """
        n = len(smoothed_probs)
        change_points = []
        
        if n <= window_size*2:
            return change_points
        
        # 添加趋势检测
        trends = []  # 存储每个点的趋势
        
        # 计算每个点的短期趋势（斜率）
        for i in range(window_size, n - window_size):
            before_window = smoothed_probs[i-window_size:i]
            after_window = smoothed_probs[i:i+window_size]
            
            # 使用简单线性回归计算斜率
            x_before = np.arange(len(before_window))
            x_after = np.arange(len(after_window))
            
            # 计算前后窗口的斜率
            if len(before_window) >= 2:
                slope_before = np.polyfit(x_before, before_window, 1)[0]
            else:
                slope_before = 0
                
            if len(after_window) >= 2:
                slope_after = np.polyfit(x_after, after_window, 1)[0]
            else:
                slope_after = 0
            
            # 保存趋势变化
            trends.append((slope_before, slope_after))
            
            # 检测趋势变化
            trend_change = slope_after - slope_before
            
            # 检测均值跳变
            mean_before = np.mean(before_window)
            mean_after = np.mean(after_window)
            mean_jump = mean_after - mean_before
            
            # 同时考虑趋势变化和均值跳变
            if (abs(trend_change) > 0.05 and abs(mean_jump) > jump_threshold):
                change_points.append(i)
            elif mean_jump > jump_threshold * 1.5:  # 如果均值跳变非常显著
                change_points.append(i)
        
        # 合并相邻的变化点
        if not change_points:
            return change_points
            
        merged_points = [change_points[0]]
        for point in change_points[1:]:
            if point - merged_points[-1] > 5:  # 至少间隔5个时间步
                merged_points.append(point)
        
        return merged_points
    
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
        try:
            # 非常短的序列特殊处理
            if len(probabilities) < 3:
                return {
                    'filtered_probs': np.array(probabilities),
                    'change_points': [],
                    'event_intervals': [(0, len(probabilities)-1)] if np.mean(probabilities) > 0.5 else []
                }
            
            # 首先对概率序列进行平滑
            smoothed_probs = self._smooth_probabilities(probabilities)
            
            # 确保平滑后的概率序列长度与原序列相同
            if len(smoothed_probs) != len(probabilities):
                print(f"警告: 平滑后长度({len(smoothed_probs)})与原长度({len(probabilities)})不一致，进行截断")
                if len(smoothed_probs) > len(probabilities):
                    smoothed_probs = smoothed_probs[:len(probabilities)]
                else:
                    padding = np.full(len(probabilities) - len(smoothed_probs), smoothed_probs[-1])
                    smoothed_probs = np.concatenate([smoothed_probs, padding])
            
            # 状态识别：首先识别序列中的稳定状态区间
            n = len(smoothed_probs)
            
            # 应用时序感知的变点检测
            
            # 1. 使用贝叶斯在线变点检测（已考虑时序相关性）
            run_length_dist = np.zeros((n, n))
            run_length_dist[0, 0] = 1
            
            change_points_bocd = []
            change_point_probs = np.zeros(n)
            
            # 在线变点检测的主循环
            for t in range(1, n):
                try:
                    # 计算风险函数
                    hazard = np.array([self.hazard_func(r) for r in range(t)])
                    
                    # 计算观测似然（已修改为时序感知的似然函数）
                    pred_history = smoothed_probs[:t]
                    likelihood = np.array([
                        self.observation_likelihood(smoothed_probs[t], pred_history, r) 
                        for r in range(t)
                    ], dtype=float)  # 强制转换为float类型
                    
                    # 确保没有None值
                    likelihood = np.nan_to_num(likelihood, nan=0.5)
                    
                    # 计算增长概率
                    growth_probs = run_length_dist[t-1, :t] * (1 - hazard) * likelihood
                    
                    # 计算变点概率
                    cp_prob = np.sum(run_length_dist[t-1, :t] * hazard * likelihood)
                    
                    # 更新运行长度分布
                    run_length_dist[t, 1:t+1] = growth_probs
                    run_length_dist[t, 0] = cp_prob
                    
                    # 归一化
                    sum_prob = np.sum(run_length_dist[t, :t+1])
                    if sum_prob > 1e-9:  # 避免除以接近零的值
                        run_length_dist[t, :t+1] /= sum_prob
                    else:
                        # 如果概率和接近零，重置为均匀分布
                        run_length_dist[t, :t+1] = 1.0 / (t+1)
                    
                    # 存储变点概率
                    change_point_probs[t] = run_length_dist[t, 0]
                    
                    # 检测变点
                    if run_length_dist[t, 0] > self.threshold:
                        # 确保与前一个变点有足够的间隔
                        if not change_points_bocd or (t - change_points_bocd[-1]) > 5:
                            change_points_bocd.append(t)
                
                except Exception as e:
                    print(f"变点检测循环异常 (t={t}): {str(e)}")
                    # 跳过这一步，继续下一个时间点的计算
                    continue
            
            # 2. 检测趋势变化和跳变
            try:
                change_points_trend = self._detect_trends_and_jumps(smoothed_probs, window_size=3, jump_threshold=0.15)
            except Exception as e:
                print(f"趋势检测异常: {str(e)}")
                change_points_trend = []
            
            # 合并两种方法的结果（去重并排序）
            all_change_points = sorted(list(set(change_points_bocd + change_points_trend)))
            
            # 如果仍未检测到变点，但存在明显的概率变化趋势，尝试强制添加变点
            if not all_change_points:
                try:
                    # 检测序列是否有明显的状态变化区间
                    
                    # 1. 使用分段算法识别不同稳定状态
                    segments = []
                    current_seg_start = 0
                    mean_levels = []
                    
                    # 简单分段：检测均值显著变化的点
                    window = 3
                    for i in range(window, n - window):
                        before_mean = np.mean(smoothed_probs[i-window:i])
                        after_mean = np.mean(smoothed_probs[i:i+window])
                        
                        if abs(after_mean - before_mean) > 0.2:
                            # 找到一个可能的分段点
                            segments.append((current_seg_start, i))
                            mean_levels.append(np.mean(smoothed_probs[current_seg_start:i]))
                            current_seg_start = i
                    
                    # 添加最后一个分段
                    if current_seg_start < n-1:
                        segments.append((current_seg_start, n-1))
                        mean_levels.append(np.mean(smoothed_probs[current_seg_start:]))
                    
                    # 如果找到了多个分段，检查是否有显著的概率水平变化
                    if len(segments) > 1:
                        for i in range(1, len(segments)):
                            prev_level = mean_levels[i-1]
                            curr_level = mean_levels[i]
                            
                            if abs(curr_level - prev_level) > 0.2:
                                # 在分段点处添加变点
                                all_change_points.append(segments[i][0])
                                print(f"在分段点处添加变点: {segments[i][0]}, 前段均值={prev_level:.2f}, 后段均值={curr_level:.2f}")
                    
                    # 2. 如果分段算法未找到变点，退化为三等分检测法
                    if not all_change_points:
                        third = len(smoothed_probs) // 3
                        first_third_mean = np.mean(smoothed_probs[:third])
                        last_two_thirds_mean = np.mean(smoothed_probs[third:])
                        
                        if abs(last_two_thirds_mean - first_third_mean) > 0.2:
                            all_change_points.append(third)
                            print(f"强制添加变点: 前1/3平均={first_third_mean:.2f}, 后2/3平均={last_two_thirds_mean:.2f}")
                except Exception as e:
                    print(f"强制添加变点异常: {str(e)}")
            
            # 识别火灾事件区间（考虑时序连续性）
            try:
                event_intervals = self._identify_fire_events_with_trends(smoothed_probs, all_change_points)
            except Exception as e:
                print(f"事件区间识别异常: {str(e)}")
                # 回退到简单的阈值检测
                event_intervals = []
                in_event = False
                start_idx = 0
                
                for i, prob in enumerate(smoothed_probs):
                    if not in_event and prob > 0.5:
                        start_idx = i
                        in_event = True
                    elif in_event and prob < 0.4:
                        event_intervals.append((start_idx, i))
                        in_event = False
                
                if in_event:
                    event_intervals.append((start_idx, len(smoothed_probs)-1))
            
            return {
                'filtered_probs': smoothed_probs,
                'change_points': all_change_points,
                'event_intervals': event_intervals,
                'change_point_probs': change_point_probs
            }
        
        except Exception as e:
            # 捕获所有未处理异常
            import traceback
            error_msg = f"BOCD检测过程中出现未处理异常: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # 返回一个安全的默认结果
            return {
                'filtered_probs': np.array(probabilities),
                'change_points': [],
                'event_intervals': [(0, len(probabilities)-1)] if np.mean(probabilities) > 0.5 else [],
                'change_point_probs': np.zeros(len(probabilities))
            }
    
    def _smooth_probabilities(self, probabilities):
        """使用滑动窗口平滑概率序列
        
        Args:
            probabilities: 原始概率序列
        
        Returns:
            平滑后的概率序列
        """
        # 如果序列长度小于窗口大小，则调整窗口大小
        window_size = min(self.smoothing_window, len(probabilities))
        if window_size < 2:
            return np.array(probabilities)  # 窗口太小，不进行平滑
        
        # 使用卷积进行平滑，确保输出长度与输入相同
        smoothed = np.convolve(
            probabilities, 
            np.ones(window_size)/window_size,
            mode='same'
        )
        
        # 处理边界情况
        pad_size = window_size // 2
        for i in range(pad_size):
            if i < len(probabilities):
                # 前边界
                window = probabilities[:i+pad_size+1]
                if len(window) > 0:
                    smoothed[i] = np.mean(window)
            
            if i < len(probabilities) and (len(probabilities) - i - 1) >= 0:
                # 后边界
                window = probabilities[-(i+pad_size+1):]
                if len(window) > 0:
                    smoothed[-(i+1)] = np.mean(window)
        
        return smoothed
    
    def _identify_fire_events_with_trends(self, probabilities, change_points):
        """根据变点和概率趋势识别火灾事件区间
        
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
        
        # 考虑序列起始点、变点和终点进行检查
        points_to_check = [0] + sorted(change_points) + [len(probabilities)-1]
        
        for i in range(len(points_to_check)-1):
            current_point = points_to_check[i]
            next_point = points_to_check[i+1]
            
            # 检查这段区间的概率特征
            segment_probs = probabilities[current_point:next_point+1]
            
            # 计算均值和趋势
            mean_prob = np.mean(segment_probs)
            
            # 趋势判断（是上升、下降还是平稳）
            if len(segment_probs) >= 3:
                x = np.arange(len(segment_probs))
                slope = np.polyfit(x, segment_probs, 1)[0]
                
                # 根据均值和趋势判断区间状态
                if not event_ongoing:
                    # 未处于事件中，判断是否应开始事件
                    if mean_prob > 0.5 or (mean_prob > 0.4 and slope > 0.05):
                        # 均值大于0.5，或均值较大且有明显上升趋势，认为是事件开始
                        start_idx = current_point
                        event_ongoing = True
                else:
                    # 已处于事件中，判断是否应结束事件
                    if mean_prob < 0.4 or (mean_prob < 0.5 and slope < -0.05):
                        # 均值小于0.4，或均值较小且有明显下降趋势，认为是事件结束
                        intervals.append((start_idx, current_point))
                        event_ongoing = False
            else:
                # 区间太短，仅使用均值判断
                if not event_ongoing and mean_prob > 0.5:
                    start_idx = current_point
                    event_ongoing = True
                elif event_ongoing and mean_prob < 0.45:
                    intervals.append((start_idx, current_point))
                    event_ongoing = False
        
        # 处理最后一个未闭合的区间
        if event_ongoing:
            intervals.append((start_idx, len(probabilities)-1))
        
        # 合并非常接近的区间（例如间隔小于3个时间步）
        if len(intervals) > 1:
            merged_intervals = [intervals[0]]
            for current_start, current_end in intervals[1:]:
                prev_start, prev_end = merged_intervals[-1]
                
                if current_start - prev_end <= 3:  # 间隔小于等于3个时间步
                    # 合并区间
                    merged_intervals[-1] = (prev_start, current_end)
                else:
                    merged_intervals.append((current_start, current_end))
            
            return merged_intervals
        
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
        
        # 确保原始概率和平滑概率长度一致
        if len(probabilities) != len(smoothed_probs):
            print(f"警告: 原始概率长度({len(probabilities)})和平滑概率长度({len(smoothed_probs)})不一致")
            # 如果长度不一致，则使用相同长度的数据进行绘图
            min_len = min(len(probabilities), len(smoothed_probs))
            probabilities = probabilities[:min_len]
            smoothed_probs = smoothed_probs[:min_len]
        
        # 创建时间轴
        time_axis = np.arange(len(probabilities)) * segment_duration
        
        plt.figure(figsize=(12, 6))
        
        # 计算变点概率
        n = len(probabilities)
        run_length_dist = np.zeros((n, n))
        run_length_dist[0, 0] = 1
        change_point_probs = np.zeros(n)
        
        # 计算每个时间点的变点概率
        for t in range(1, n):
            hazard = np.array([self._constant_hazard(r) for r in range(t)])
            pred_history = smoothed_probs[:t]
            likelihood = np.array([
                self._trend_aware_likelihood(smoothed_probs[t], pred_history, r) 
                for r in range(t)
            ])
            
            growth_probs = run_length_dist[t-1, :t] * (1 - hazard) * likelihood
            cp_prob = np.sum(run_length_dist[t-1, :t] * hazard * likelihood)
            
            run_length_dist[t, 1:t+1] = growth_probs
            run_length_dist[t, 0] = cp_prob
            
            run_length_dist[t, :t+1] /= np.sum(run_length_dist[t, :t+1]) + 1e-9
            change_point_probs[t] = run_length_dist[t, 0]
        
        # 根据阈值绘制不同颜色的变点概率
        below_threshold = change_point_probs < self.threshold
        above_threshold = ~below_threshold
        
        # 绘制低于阈值的点（绿色）
        if np.any(below_threshold):
            plt.plot(time_axis[below_threshold], change_point_probs[below_threshold], 
                    'g-', label=f'变点概率 < {self.threshold}')
        
        # 绘制高于阈值的点（红色）
        if np.any(above_threshold):
            plt.plot(time_axis[above_threshold], change_point_probs[above_threshold], 
                    'r-', label=f'变点概率 ≥ {self.threshold}')
        
        # 绘制阈值线
        plt.axhline(y=self.threshold, color='k', linestyle='--', alpha=0.5, 
                   label=f'阈值 ({self.threshold})')
        
        plt.grid(True)
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('变点概率', fontsize=12)
        plt.title('火灾事件变点检测结果', fontsize=14)
        plt.legend(fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        
        return plt.gcf() 