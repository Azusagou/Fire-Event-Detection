import os
import h5py
import numpy as np
import pandas as pd
import tqdm
import librosa

np.random.seed(42)

def parse_seconds(string):
    """将时间字符串转换为秒数
    
    Args:
        string: 格式为"MM:SS"的时间字符串
        
    Returns:
        int: 转换后的秒数
    """
    minutes, seconds = [int(s) for s in string.split(":")]
    return minutes * 60 + seconds

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

    #print("wave: ", wave.shape)
    #print("wave_size: ", wave_size)
    #print("segment_size: ", segment_size)
    nb_remove = wave_size % segment_size
    if nb_remove > 0:
        truncated_wave = wave[:-nb_remove]
    else:
        truncated_wave = wave

    if not truncated_wave.shape[0] % segment_size == 0:
       raise ValueError("reapeated wave not even multiple of segment size")

    nb_segments = int(truncated_wave.shape[0]/segment_size)
    #print("truncated_wave: ", truncated_wave.shape)
    #print("nb_segments: ", nb_segments)
    segments = np.split(truncated_wave, nb_segments, axis=0)

    return segments

def prepare_fire_data(source_dir, csv_file, segment_time, hdf5_path, sample_rate):    
    """准备火灾事件音频数据集
    
    读取原始音频文件，进行预处理和分割，并保存为HDF5格式
    
    Args:
        source_dir: 原始音频文件目录
        csv_file: 数据标注文件路径
        segment_time: 音频片段时长
        hdf5_path: 输出HDF5文件路径
        sample_rate: 目标采样率
    """
    # 读取数据标注文件
    df = pd.read_csv(csv_file)
    
    # 检查所有音频文件是否存在
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df.index)):
        filename = row[0]
        start_time = row[1]
        end_time = row[2]
        fire_event = bool(int(row[3]))
        if not os.path.exists(os.path.join(source_dir, filename + ".WAV")):
            raise ValueError("file does not exist: ", filename)
    
    # 第一遍：计算数据统计信息
    nb_rows = 0
    sum_mean = 0
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df.index)):
        filename = row[0]
        start_time = row[1]
        end_time = row[2]
        fire_event = bool(int(row[3]))

        #print("------------------------------------------")
        #print("file_name: ", filename)
        #print("start_time: ", start_time)
        #print("end_time:", end_time)
        wave, sample_rate = librosa.load(os.path.join(source_dir, filename) + ".WAV", sr=sample_rate, mono=True, res_type='kaiser_fast')
        start_second = parse_seconds(start_time)
        end_second = parse_seconds(end_time)
        #print("start_second: ", start_second)
        #print("end_second: ", end_second)
        wave_preprocesed = wave[start_second*sample_rate:end_second*sample_rate]
        segments = split_into_segments(wave_preprocesed, sample_rate=sample_rate, segment_time=segment_time)

        nb_rows += len(segments)
        sum_mean += np.sum(np.concatenate(segments))
        
    nb_columns = segment_time * sample_rate
    
    n = nb_rows * nb_columns
    mean = sum_mean / n
    
    # 设置数据集划分比例
    train_fraction = 0.7
    valid_fraction = 0.1
    test_fraction = 0.2
    
    # 创建HDF5文件并设置数据集
    with h5py.File(hdf5_path, 'w') as f:
        print("dataset shape: ({}, {})".format(nb_rows, nb_columns))
        # 创建数据集
        dset_x = f.create_dataset("wave_segments", (nb_rows, nb_columns), 'f')
        dset_y = f.create_dataset("class_labels", (nb_rows, 1), 'i')
        dset_segment_indices = f.create_dataset("file_segment_indices", (nb_rows, 1), 'i')
        dset_csv_row_indices = f.create_dataset("csv_row_indices", (nb_rows, 1))
        dset_stats = f.create_dataset("statistics", (1, 3), 'f')

        # 随机划分训练/验证/测试集
        indices = np.arange(nb_rows)
        np.random.shuffle(indices)
        split_1 = int(nb_rows*train_fraction)
        split_2 = int(nb_rows*(train_fraction+valid_fraction))
        train_indices, valid_indices, test_indices = np.split(indices, [split_1, split_2])

        # 保存数据集划分索引
        dset_train_indices = f.create_dataset("train_indices", (len(train_indices),), 'i')
        dset_valid_indices = f.create_dataset("valid_indices", (len(valid_indices),), 'i')
        dset_test_indices  = f.create_dataset("test_indices",  (len(test_indices),), 'i')

        dset_train_indices[:] = train_indices
        dset_valid_indices[:] = valid_indices
        dset_test_indices[:]  = test_indices

        # 第二遍：计算方差并填充数据集
        sum_variance = 0
        running_idx = 0
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df.index)):
            filename = row[0]
            start_time = row[1]
            end_time = row[2]
            fire_event = int(row[3])

            wave, sample_rate = librosa.load(os.path.join(source_dir, filename) + ".WAV", sr=sample_rate, mono=True, res_type='kaiser_fast')
            start_second = parse_seconds(start_time)
            end_second = parse_seconds(end_time)
            wave_preprocesed = wave[start_second*sample_rate:end_second*sample_rate]
            segments = split_into_segments(wave_preprocesed, sample_rate=sample_rate, segment_time=segment_time)

            sum_variance += np.sum(np.power((np.concatenate(segments)-mean), 2))

            # 填充数据集
            for i in range(len(segments)):
                dset_x[running_idx, :] = segments[i]
                dset_y[running_idx, :] = fire_event
                dset_segment_indices[running_idx, :] = i
                dset_csv_row_indices[running_idx, :] = index
                running_idx += 1

        # 保存数据集统计信息
        variance = sum_variance / (n-1)
        dset_stats[0,:] = [mean, variance, sample_rate]

def main():
    """主函数"""
    # 设置参数
    source_dir = "./wav"  # 原始音频文件目录
    dataset_name = "spruce_oak_pmma_pur_chipboard"  # 数据集名称
    csv_file = dataset_name + ".csv"  # 数据标注文件
    sample_rate = 32000  # 采样率
    segment_length = 5   # 音频片段时长(秒)

    # 生成HDF5文件路径
    hdf5_path = "dataset_{}_sr_{}.hdf5".format(dataset_name, sample_rate)
    if not os.path.exists(hdf5_path):
        prepare_fire_data(source_dir, csv_file, segment_length, hdf5_path, sample_rate)

if __name__ == '__main__':
    main()
