"""
EDF 腦波訊號時頻域分析程式
使用短時傅立葉變換（STFT）處理 EEG 訊號

作者: Wayne
日期: 2026-02-04
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import struct
from pathlib import Path

# 設定中文字體（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1部分：自己實現 EDF 讀取（不依賴 pyedflib）
# ============================================================

def read_edf(filepath):
    """
    讀取 EDF 或 BDF 檔案並返回訊號資料
    
    自動檢測檔案格式：
    - EDF: European Data Format (16-bit)
    - BDF: BioSemi Data Format (24-bit)
    
    參數:
        filepath: EDF/BDF 檔案路徑
    
    返回:
        signals: 各通道訊號數據 (list of arrays)
        signal_headers: 各通道的標頭資訊
        header: 檔案總標頭資訊
    """
    with open(filepath, 'rb') as f:
        # === 讀取主標頭 (256 bytes) ===
        header = {}
        version_bytes = f.read(8)
        
        # 檢測檔案格式：BDF 的版本號第一個 byte 為 0xFF
        is_bdf = (version_bytes[0] == 255)
        header['format'] = 'BDF' if is_bdf else 'EDF'
        header['bit_depth'] = 24 if is_bdf else 16
        
        # BDF: 跳過第一個 0xFF byte，解碼剩餘部分
        # EDF: 正常解碼整個版本號
        if is_bdf:
            header['version'] = version_bytes[1:].decode('ascii').strip()
        else:
            header['version'] = version_bytes.decode('ascii').strip()
        
        header['patient'] = f.read(80).decode('ascii').strip()
        header['recording'] = f.read(80).decode('ascii').strip()
        header['startdate'] = f.read(8).decode('ascii').strip()
        header['starttime'] = f.read(8).decode('ascii').strip()
        header['header_bytes'] = int(f.read(8).decode('ascii').strip())
        header['reserved'] = f.read(44).decode('ascii').strip()
        header['n_records'] = int(f.read(8).decode('ascii').strip())
        header['record_duration'] = float(f.read(8).decode('ascii').strip())
        header['n_channels'] = int(f.read(4).decode('ascii').strip())
        
        n_channels = header['n_channels']
        
        # === 讀取各通道標頭 (每通道 256 bytes) ===
        signal_headers = []
        labels = [f.read(16).decode('ascii').strip() for _ in range(n_channels)]
        transducers = [f.read(80).decode('ascii').strip() for _ in range(n_channels)]
        dimensions = [f.read(8).decode('ascii').strip() for _ in range(n_channels)]
        physical_mins = [float(f.read(8).decode('ascii').strip()) for _ in range(n_channels)]
        physical_maxs = [float(f.read(8).decode('ascii').strip()) for _ in range(n_channels)]
        digital_mins = [int(f.read(8).decode('ascii').strip()) for _ in range(n_channels)]
        digital_maxs = [int(f.read(8).decode('ascii').strip()) for _ in range(n_channels)]
        prefilterings = [f.read(80).decode('ascii').strip() for _ in range(n_channels)]
        n_samples = [int(f.read(8).decode('ascii').strip()) for _ in range(n_channels)]
        reserveds = [f.read(32).decode('ascii').strip() for _ in range(n_channels)]
        
        for i in range(n_channels):
            signal_headers.append({
                'label': labels[i],
                'transducer': transducers[i],
                'dimension': dimensions[i],
                'physical_min': physical_mins[i],
                'physical_max': physical_maxs[i],
                'digital_min': digital_mins[i],
                'digital_max': digital_maxs[i],
                'prefiltering': prefilterings[i],
                'n_samples': n_samples[i],
                'sample_rate': n_samples[i] / header['record_duration']
            })
        
        # === 讀取數據記錄 ===
        signals = [[] for _ in range(n_channels)]
        
        if is_bdf:
            # BDF: 24-bit 整數
            for _ in range(header['n_records']):
                for ch in range(n_channels):
                    # 讀取 24-bit 整數數據（每個樣本 3 bytes）
                    raw_data = f.read(n_samples[ch] * 3)
                    digital_values = []
                    
                    # 將 3-byte 轉換為帶符號的 24-bit 整數
                    for i in range(n_samples[ch]):
                        # Little-endian: 低位元組在前
                        byte1 = raw_data[i * 3]
                        byte2 = raw_data[i * 3 + 1]
                        byte3 = raw_data[i * 3 + 2]
                        
                        # 組合成 24-bit 整數
                        value = byte1 | (byte2 << 8) | (byte3 << 16)
                        
                        # 處理符號位（如果第 23 位為 1，代表負數）
                        if value & 0x800000:  # 檢查符號位
                            value = value - 0x1000000  # 轉換為負數
                        
                        digital_values.append(value)
                    
                    # 將數位值轉換為物理值
                    scale = (physical_maxs[ch] - physical_mins[ch]) / \
                            (digital_maxs[ch] - digital_mins[ch])
                    offset = physical_mins[ch] - digital_mins[ch] * scale
                    
                    physical_values = [d * scale + offset for d in digital_values]
                    signals[ch].extend(physical_values)
        else:
            # EDF: 16-bit 整數
            for _ in range(header['n_records']):
                for ch in range(n_channels):
                    # 讀取 16-bit 整數數據
                    raw_data = f.read(n_samples[ch] * 2)
                    digital_values = struct.unpack(f'<{n_samples[ch]}h', raw_data)
                    
                    # 將數位值轉換為物理值
                    scale = (physical_maxs[ch] - physical_mins[ch]) / \
                            (digital_maxs[ch] - digital_mins[ch])
                    offset = physical_mins[ch] - digital_mins[ch] * scale
                    
                    physical_values = [d * scale + offset for d in digital_values]
                    signals[ch].extend(physical_values)
        
        # 轉換為 numpy array
        signals = [np.array(s) for s in signals]
        
    return signals, signal_headers, header


# ============================================================
# 第2部分：訊號預處理（去雜訊）
# ============================================================

def bandpass_filter(signal_data, fs, low_freq=0.5, high_freq=50, order=4):
    """
    帶通濾波器 - 去除雜訊
    
    參數:
        signal_data: 輸入訊號
        fs: 採樣頻率 (Hz)
        low_freq: 低截止頻率 (Hz)，去除基線漂移
        high_freq: 高截止頻率 (Hz)，去除高頻雜訊
        order: 濾波器階數
    
    返回:
        filtered_signal: 濾波後的訊號
    """
    # 計算正規化頻率 (相對於 Nyquist 頻率)
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # 確保頻率在有效範圍內
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        print(f"警告: 濾波頻率範圍無效 ({low_freq}-{high_freq} Hz)，跳過濾波")
        return signal_data
    
    # 設計 Butterworth 帶通濾波器
    b, a = signal.butter(order, [low, high], btype='band')
    
    # 使用 filtfilt 進行零相位濾波（前後各濾一次，消除相位延遲）
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal


def notch_filter(signal_data, fs, freq=50, Q=30):
    """
    陷波濾波器 - 去除工頻干擾 (50/60 Hz)
    
    參數:
        signal_data: 輸入訊號
        fs: 採樣頻率 (Hz)
        freq: 要去除的頻率 (Hz)
        Q: 品質因子，越大帶寬越窄
    
    返回:
        filtered_signal: 濾波後的訊號
    """
    nyquist = fs / 2
    if freq >= nyquist:
        return signal_data
    
    # 設計陷波濾波器
    b, a = signal.iirnotch(freq / nyquist, Q)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal


def preprocess_eeg(signal_data, fs, bandpass=(0.5, 50), notch=None):
    """
    EEG 訊號預處理流程
    
    參數:
        signal_data: 原始訊號
        fs: 採樣頻率
        bandpass: (低頻, 高頻) 帶通濾波範圍，None 表示不濾波
        notch: 陷波頻率（50 或 60 Hz），None 表示不濾波
    
    返回:
        processed_signal: 預處理後的訊號
    """
    processed = signal_data.copy()
    
    # 1. 帶通濾波
    if bandpass is not None:
        low, high = bandpass
        processed = bandpass_filter(processed, fs, low, high)
    
    # 2. 陷波濾波（去除工頻干擾）
    if notch is not None:
        processed = notch_filter(processed, fs, notch)
    
    return processed


# ============================================================
# 第3部分：短時傅立葉變換（STFT）
# ============================================================

def custom_stft(signal_data, fs, window_size=256, hop_size=64, window_type='hann'):
    """
    自己實現短時傅立葉變換
    
    參數:
        signal_data: 輸入訊號（1D array）
        fs: 採樣頻率 (Hz)
        window_size: 視窗大小（點數）
        hop_size: 跳躍步長（點數）
        window_type: 視窗類型 ('hann', 'hamming', 'blackman', 'rect')
    
    返回:
        time_axis: 時間軸
        freq_axis: 頻率軸
        Zxx: 時頻矩陣（複數）
    """
    n_samples = len(signal_data)
    
    # 選擇視窗函數
    if window_type == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1)))
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    elif window_type == 'blackman':
        window = 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1)) \
                 + 0.08 * np.cos(4 * np.pi * np.arange(window_size) / (window_size - 1))
    else:
        window = np.ones(window_size)  # 矩形窗
    
    # 計算視窗數量
    n_windows = 1 + (n_samples - window_size) // hop_size
    
    # 初始化輸出矩陣
    n_freqs = window_size // 2 + 1
    Zxx = np.zeros((n_freqs, n_windows), dtype=complex)
    
    # 執行 STFT
    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        
        # 擷取訊號片段並加窗
        segment = signal_data[start:end] * window
        
        # 執行 FFT 並取正頻率部分
        fft_result = np.fft.fft(segment)
        Zxx[:, i] = fft_result[:n_freqs]
    
    # 計算時間軸和頻率軸
    time_axis = (np.arange(n_windows) * hop_size + window_size / 2) / fs
    freq_axis = np.fft.rfftfreq(window_size, 1 / fs)
    
    return time_axis, freq_axis, Zxx


def scipy_stft(signal_data, fs, window_size=256, hop_size=64):
    """
    使用 scipy 的 STFT（用於對照驗證）
    """
    f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=window_size, 
                            noverlap=window_size-hop_size, window='hann')
    return t, f, Zxx


# ============================================================
# 第3部分：EEG 頻帶分析
# ============================================================

EEG_BANDS = {
    'Delta': (0.5, 4),    # 深度睡眠
    'Theta': (4, 8),      # 淺睡眠、放鬆
    'Alpha': (8, 13),     # 清醒放鬆
    'Beta': (13, 30),     # 專注、思考
    'Gamma': (30, 100)    # 認知處理
}

def extract_band_power(freq_axis, Zxx, band_range):
    """
    提取特定頻帶的功率
    
    參數:
        freq_axis: 頻率軸
        Zxx: STFT 結果
        band_range: (低頻, 高頻) 頻帶範圍
    
    返回:
        band_power: 該頻帶隨時間變化的功率
    """
    low, high = band_range
    band_mask = (freq_axis >= low) & (freq_axis <= high)
    power = np.abs(Zxx) ** 2
    band_power = np.mean(power[band_mask, :], axis=0)
    return band_power


def analyze_eeg_bands(time_axis, freq_axis, Zxx):
    """
    分析各 EEG 頻帶功率
    """
    band_powers = {}
    for band_name, band_range in EEG_BANDS.items():
        band_powers[band_name] = extract_band_power(freq_axis, Zxx, band_range)
    return band_powers


def calculate_band_ratio(band_powers, band1, band2):
    """
    計算兩個頻帶的功率比值
    常用於：
    - Theta/Beta ratio: 注意力指標
    - Alpha/Beta ratio: 放鬆程度
    """
    return band_powers[band1] / (band_powers[band2] + 1e-10)


def identify_dominant_band(band_powers, time_axis):
    """
    識別每個時間點的主導頻帶
    
    參數:
        band_powers: 各頻帶功率字典 {band_name: power_array}
        time_axis: 時間軸
    
    返回:
        dominant_bands: 每個時間點的主導頻帶名稱列表
        dominant_stats: 統計資訊字典
    """
    n_times = len(time_axis)
    band_names = list(band_powers.keys())
    
    # 建立功率矩陣 (bands x time)
    power_matrix = np.array([band_powers[name] for name in band_names])
    
    # 找出每個時間點功率最大的頻帶索引
    dominant_indices = np.argmax(power_matrix, axis=0)
    
    # 轉換為頻帶名稱
    dominant_bands = [band_names[i] for i in dominant_indices]
    
    # 計算統計資訊
    dominant_stats = {
        'counts': {},       # 各頻帶主導的次數
        'percentages': {},  # 各頻帶主導的百分比
        'durations': {},    # 各頻帶主導的總時長（秒）
    }
    
    dt = time_axis[1] - time_axis[0] if len(time_axis) > 1 else 0
    
    for band_name in band_names:
        count = dominant_bands.count(band_name)
        dominant_stats['counts'][band_name] = count
        dominant_stats['percentages'][band_name] = count / n_times * 100
        dominant_stats['durations'][band_name] = count * dt
    
    # 找出整體最主導的頻帶
    dominant_stats['overall_dominant'] = max(
        dominant_stats['counts'], 
        key=dominant_stats['counts'].get
    )
    
    return dominant_bands, dominant_stats


def get_band_color(band_name):
    """取得各頻帶的代表顏色"""
    colors = {
        'Delta': '#1f77b4',   # 藍色
        'Theta': '#ff7f0e',   # 橙色
        'Alpha': '#2ca02c',   # 綠色
        'Beta': '#d62728',    # 紅色
        'Gamma': '#9467bd'    # 紫色
    }
    return colors.get(band_name, '#333333')


def plot_dominant_bands(time_axis, dominant_bands, dominant_stats, 
                        title="主導頻帶識別", save_path=None):
    """
    繪製主導頻帶隨時間變化的圖表
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # === 上圖：主導頻帶時間序列 ===
    ax1 = axes[0]
    band_names = list(EEG_BANDS.keys())
    
    # 為每個時間點繪製對應頻帶的顏色
    for i, (t, band) in enumerate(zip(time_axis, dominant_bands)):
        color = get_band_color(band)
        ax1.axvspan(t - (time_axis[1]-time_axis[0])/2 if i > 0 else t, 
                    t + (time_axis[1]-time_axis[0])/2, 
                    alpha=0.7, color=color, linewidth=0)
    
    ax1.set_xlim([time_axis[0], time_axis[-1]])
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('主導頻帶')
    ax1.set_title(title)
    
    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=get_band_color(b), label=f'{b}') 
                       for b in band_names]
    ax1.legend(handles=legend_elements, loc='upper right', ncol=5)
    
    # === 下圖：統計長條圖 ===
    ax2 = axes[1]
    percentages = [dominant_stats['percentages'][b] for b in band_names]
    colors = [get_band_color(b) for b in band_names]
    
    bars = ax2.bar(band_names, percentages, color=colors, alpha=0.8)
    ax2.set_ylabel('主導時間 (%)')
    ax2.set_ylim([0, 100])
    
    # 在長條上顯示數值
    for bar, pct in zip(bars, percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 標記整體最主導的頻帶
    overall = dominant_stats['overall_dominant']
    ax2.set_title(f"整體主導頻帶: {overall} ({dominant_stats['percentages'][overall]:.1f}%)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    plt.show()


# ============================================================
# 第5部分：視覺化
# ============================================================

def plot_spectrogram(time_axis, freq_axis, Zxx, title="EEG 時頻譜圖", 
                     freq_limit=50, save_path=None):
    """
    繪製時頻譜圖
    """
    plt.figure(figsize=(14, 6))
    
    # 計算功率譜（dB）
    power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
    
    # 限制頻率範圍
    freq_mask = freq_axis <= freq_limit
    
    plt.pcolormesh(time_axis, freq_axis[freq_mask], power_db[freq_mask, :], 
                   shading='gouraud', cmap='jet')
    plt.colorbar(label='功率 (dB)')
    plt.ylabel('頻率 (Hz)')
    plt.xlabel('時間 (秒)')
    plt.title(title)
    
    # 標註 EEG 頻帶分界線
    for band_name, (low, high) in EEG_BANDS.items():
        if high <= freq_limit:
            plt.axhline(y=low, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
            plt.text(time_axis[0] + 0.5, (low + high) / 2, band_name, 
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    plt.show()


def plot_band_powers(time_axis, band_powers, title="EEG 各頻帶功率", save_path=None):
    """
    繪製各頻帶功率隨時間的變化
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, axes = plt.subplots(len(band_powers), 1, figsize=(14, 12), sharex=True)
    
    for ax, (band_name, power), color in zip(axes, band_powers.items(), colors):
        ax.fill_between(time_axis, power, alpha=0.3, color=color)
        ax.plot(time_axis, power, color=color, linewidth=1)
        ax.set_ylabel(f'{band_name}\n功率', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_axis[0], time_axis[-1]])
        
        # 顯示頻率範圍
        freq_range = EEG_BANDS[band_name]
        ax.text(0.98, 0.95, f'{freq_range[0]}-{freq_range[1]} Hz', 
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='gray')
    
    axes[-1].set_xlabel('時間 (秒)', fontsize=11)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    plt.show()


def plot_raw_signal(signal_data, fs, duration=10, title="原始 EEG 訊號", save_path=None):
    """
    繪製原始訊號波形
    
    參數:
        signal_data: 訊號數據
        fs: 採樣頻率
        duration: 顯示時長（秒），None 表示全部
    """
    if duration:
        n_samples = int(duration * fs)
        signal_data = signal_data[:n_samples]
    
    time_axis = np.arange(len(signal_data)) / fs
    
    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, signal_data, linewidth=0.5, color='#1f77b4')
    plt.xlabel('時間 (秒)')
    plt.ylabel('振幅 (μV)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    plt.show()


def plot_summary(time_axis, freq_axis, Zxx, band_powers, signal_data, fs,
                 title="EEG 分析總覽", save_path=None):
    """
    繪製綜合分析圖
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 原始訊號（取前10秒）
    ax1 = plt.subplot(3, 1, 1)
    duration = min(10, len(signal_data) / fs)
    n_samples = int(duration * fs)
    t_raw = np.arange(n_samples) / fs
    ax1.plot(t_raw, signal_data[:n_samples], linewidth=0.5, color='#1f77b4')
    ax1.set_ylabel('振幅 (μV)')
    ax1.set_title('原始 EEG 訊號（前 10 秒）')
    ax1.grid(True, alpha=0.3)
    
    # 2. 時頻譜圖
    ax2 = plt.subplot(3, 1, 2)
    power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
    freq_mask = freq_axis <= 50
    im = ax2.pcolormesh(time_axis, freq_axis[freq_mask], power_db[freq_mask, :], 
                        shading='gouraud', cmap='jet')
    plt.colorbar(im, ax=ax2, label='功率 (dB)')
    ax2.set_ylabel('頻率 (Hz)')
    ax2.set_title('時頻譜圖 (STFT)')
    
    # 3. 各頻帶功率堆疊圖
    ax3 = plt.subplot(3, 1, 3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 正規化功率用於堆疊顯示
    normalized_powers = {}
    total_power = sum(band_powers.values())
    for band_name, power in band_powers.items():
        normalized_powers[band_name] = power / (total_power + 1e-10) * 100
    
    bottom = np.zeros(len(time_axis))
    for (band_name, power), color in zip(normalized_powers.items(), colors):
        ax3.fill_between(time_axis, bottom, bottom + power, 
                        alpha=0.7, color=color, label=band_name)
        bottom += power
    
    ax3.set_xlabel('時間 (秒)')
    ax3.set_ylabel('相對功率 (%)')
    ax3.set_title('各頻帶相對功率變化')
    ax3.legend(loc='upper right', ncol=5)
    ax3.set_xlim([time_axis[0], time_axis[-1]])
    ax3.set_ylim([0, 100])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    plt.show()


# ============================================================
# 第5部分：主程式
# ============================================================

def process_edf_file(edf_path, channel_index=0, output_dir=None):
    """
    處理 EDF/BDF 檔案的主函數
    
    參數:
        edf_path: EDF/BDF 檔案路徑
        channel_index: 要分析的通道索引
        output_dir: 輸出目錄（儲存圖片），None 表示不儲存
    """
    edf_path = Path(edf_path)
    print(f"正在讀取檔案: {edf_path}")
    
    # 讀取 EDF/BDF
    signals, signal_headers, header = read_edf(edf_path)
    
    print(f"\n{'='*50}")
    print(f"{header['format']} 檔案資訊 ({header['bit_depth']}-bit)")
    print(f"{'='*50}")
    print(f"檔案格式: {header['format']}")
    print(f"位元深度: {header['bit_depth']} bits")
    print(f"患者資訊: {header['patient']}")
    print(f"錄製日期: {header['startdate']} {header['starttime']}")
    print(f"通道數量: {header['n_channels']}")
    print(f"記錄時長: {header['n_records'] * header['record_duration']:.1f} 秒")
    
    print(f"\n{'='*50}")
    print(f"通道資訊")
    print(f"{'='*50}")
    for i, sh in enumerate(signal_headers):
        print(f"  [{i}] {sh['label']}: {sh['sample_rate']:.1f} Hz, {len(signals[i])} 點")
    
    # 檢查通道索引
    if channel_index >= len(signals):
        print(f"\n警告: 通道索引 {channel_index} 超出範圍，使用通道 0")
        channel_index = 0
    
    # 選取要分析的通道
    eeg_signal = signals[channel_index]
    fs = signal_headers[channel_index]['sample_rate']
    channel_label = signal_headers[channel_index]['label']
    
    print(f"\n{'='*50}")
    print(f"分析通道: [{channel_index}] {channel_label}")
    print(f"{'='*50}")
    print(f"採樣頻率: {fs} Hz")
    print(f"訊號長度: {len(eeg_signal)} 點 ({len(eeg_signal)/fs:.1f} 秒)")
    
    # === 訊號預處理（帶通濾波去雜訊）===
    print(f"\n{'='*50}")
    print(f"訊號預處理")
    print(f"{'='*50}")
    print(f"  帶通濾波: 0.5 - 50 Hz")
    
    eeg_filtered = preprocess_eeg(eeg_signal, fs, bandpass=(0.5, 50), notch=None)
    print(f"  濾波完成！")
    
    # 執行 STFT（使用濾波後的訊號）
    window_size = int(fs * 1)      # 1 秒視窗
    hop_size = int(fs * 0.1)       # 0.1 秒步長（重疊 90%）
    
    print(f"\nSTFT 參數:")
    print(f"  視窗大小: {window_size} 點 ({window_size/fs:.1f} 秒)")
    print(f"  跳躍步長: {hop_size} 點 ({hop_size/fs:.2f} 秒)")
    print(f"  重疊率: {(1 - hop_size/window_size)*100:.0f}%")
    
    time_axis, freq_axis, Zxx = custom_stft(eeg_filtered, fs, 
                                             window_size=window_size, 
                                             hop_size=hop_size,
                                             window_type='hann')
    
    print(f"\nSTFT 結果:")
    print(f"  時間範圍: {time_axis[0]:.2f} ~ {time_axis[-1]:.2f} 秒")
    print(f"  頻率範圍: {freq_axis[0]:.2f} ~ {freq_axis[-1]:.2f} Hz")
    print(f"  頻率解析度: {freq_axis[1] - freq_axis[0]:.2f} Hz")
    print(f"  時頻矩陣大小: {Zxx.shape} (頻率 x 時間)")
    
    # 分析各頻帶
    band_powers = analyze_eeg_bands(time_axis, freq_axis, Zxx)
    
    print(f"\n{'='*50}")
    print(f"各頻帶平均功率")
    print(f"{'='*50}")
    total_power = sum(np.mean(p) for p in band_powers.values())
    for band_name, power in band_powers.items():
        mean_power = np.mean(power)
        percentage = mean_power / total_power * 100
        freq_range = EEG_BANDS[band_name]
        print(f"  {band_name:6s} ({freq_range[0]:4.1f}-{freq_range[1]:5.1f} Hz): "
              f"{mean_power:10.2f} ({percentage:5.1f}%)")
    
    # === 主導頻帶識別 ===
    dominant_bands, dominant_stats = identify_dominant_band(band_powers, time_axis)
    
    print(f"\n{'='*50}")
    print(f"主導頻帶分析")
    print(f"{'='*50}")
    print(f"  整體主導頻帶: {dominant_stats['overall_dominant']}")
    print(f"\n  各頻帶主導時間占比:")
    for band_name in EEG_BANDS.keys():
        pct = dominant_stats['percentages'][band_name]
        dur = dominant_stats['durations'][band_name]
        print(f"    {band_name:6s}: {pct:5.1f}% ({dur:.2f} 秒)")
    
    # 計算注意力指標
    theta_beta_ratio = calculate_band_ratio(band_powers, 'Theta', 'Beta')
    alpha_beta_ratio = calculate_band_ratio(band_powers, 'Alpha', 'Beta')
    print(f"\n注意力指標:")
    print(f"  Theta/Beta ratio (平均): {np.mean(theta_beta_ratio):.2f}")
    print(f"  Alpha/Beta ratio (平均): {np.mean(alpha_beta_ratio):.2f}")
    
    # 設定輸出路徑
    if output_dir is None:
        output_dir = edf_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    base_name = edf_path.stem
    
    # 繪製圖表
    print(f"\n正在繪製圖表...")
    
    # 1. 原始訊號 vs 濾波後訊號
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    duration = min(10, len(eeg_signal) / fs)
    n_samples = int(duration * fs)
    t_raw = np.arange(n_samples) / fs
    
    axes[0].plot(t_raw, eeg_signal[:n_samples], linewidth=0.5, color='#1f77b4')
    axes[0].set_ylabel('振幅 (μV)')
    axes[0].set_title(f'原始 EEG 訊號 - {channel_label}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_raw, eeg_filtered[:n_samples], linewidth=0.5, color='#2ca02c')
    axes[1].set_ylabel('振幅 (μV)')
    axes[1].set_xlabel('時間 (秒)')
    axes[1].set_title(f'濾波後訊號 (0.5-50 Hz 帶通濾波)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_filtered_signal.png", dpi=150, bbox_inches='tight')
    print(f"圖片已儲存至: {output_dir / f'{base_name}_filtered_signal.png'}")
    plt.show()
    
    # 2. 時頻譜圖
    plot_spectrogram(time_axis, freq_axis, Zxx, 
                    title=f"EEG 時頻譜圖 (濾波後) - {channel_label}",
                    save_path=output_dir / f"{base_name}_spectrogram.png")
    
    # 3. 各頻帶功率
    plot_band_powers(time_axis, band_powers,
                    title=f"EEG 各頻帶功率 - {channel_label}",
                    save_path=output_dir / f"{base_name}_band_powers.png")
    
    # 4. 主導頻帶識別圖
    plot_dominant_bands(time_axis, dominant_bands, dominant_stats,
                       title=f"主導頻帶識別 - {channel_label}",
                       save_path=output_dir / f"{base_name}_dominant_bands.png")
    
    # 5. 綜合分析圖
    plot_summary(time_axis, freq_axis, Zxx, band_powers, eeg_filtered, fs,
                title=f"EEG 分析總覽 - {channel_label}",
                save_path=output_dir / f"{base_name}_summary.png")
    
    print(f"\n分析完成！圖片已儲存至: {output_dir}")
    
    return {
        'time_axis': time_axis,
        'freq_axis': freq_axis,
        'Zxx': Zxx,
        'band_powers': band_powers,
        'dominant_bands': dominant_bands,
        'dominant_stats': dominant_stats,
        'signals': signals,
        'signal_filtered': eeg_filtered,
        'signal_headers': signal_headers,
        'header': header
    }


# 此檔案為函式庫，請使用 main.py 執行分析
