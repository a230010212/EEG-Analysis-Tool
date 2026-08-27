"""
EEG 腦波訊號分析主程式
基於 MATLAB 代碼邏輯轉譯而成

功能:
    1. 讀取 EDF/BDF 檔案
    2. 提取指定通道數據
    3. 應用帶通濾波器 (1-30 Hz)
    4. 計算 FFT
    5. 繪製時域和頻域圖

使用方式:
    python main.py              # 互動式選擇檔案
    python main.py 1            # 直接選擇第 1 個檔案
    python main.py --list       # 列出所有可用檔案

作者: Wayne
日期: 2026-02-11
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import struct

# 設定中文字體（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# EDF/BDF 檔案資料夾
# EDF/BDF 檔案資料夾
# 優先搜尋執行檔所在目錄下的 .edf 資料夾
current_dir = Path.cwd()
local_edf = current_dir / ".edf"

if local_edf.exists():
    EDF_FOLDER = local_edf
else:
    # 開發環境預設路徑 (Fallback)
    EDF_FOLDER = Path(r"D:\Wayne's Project\腦波訊號判讀小程式\.edf")

print(f"EdF 資料夾位置: {EDF_FOLDER}")


# ============================================================
# 第1部分：EDF/BDF 檔案讀取
# 對應 MATLAB: edfinfo() + edfread()
# ============================================================

def read_edf(filepath):
    """
    讀取 EDF 或 BDF 檔案並返回訊號資料
    對應 MATLAB 的 edfread 和 edfinfo 功能

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

        # === 讀取各通道標頭 ===
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
                    raw_data = f.read(n_samples[ch] * 3)
                    digital_values = []

                    for i in range(n_samples[ch]):
                        byte1 = raw_data[i * 3]
                        byte2 = raw_data[i * 3 + 1]
                        byte3 = raw_data[i * 3 + 2]

                        value = byte1 | (byte2 << 8) | (byte3 << 16)

                        if value & 0x800000:
                            value = value - 0x1000000

                        digital_values.append(value)

                    scale = (physical_maxs[ch] - physical_mins[ch]) / \
                            (digital_maxs[ch] - digital_mins[ch])
                    offset = physical_mins[ch] - digital_mins[ch] * scale

                    physical_values = [d * scale + offset for d in digital_values]
                    signals[ch].extend(physical_values)
        else:
            # EDF: 16-bit 整數
            for _ in range(header['n_records']):
                for ch in range(n_channels):
                    raw_data = f.read(n_samples[ch] * 2)
                    digital_values = struct.unpack(f'<{n_samples[ch]}h', raw_data)

                    scale = (physical_maxs[ch] - physical_mins[ch]) / \
                            (digital_maxs[ch] - digital_mins[ch])
                    offset = physical_mins[ch] - digital_mins[ch] * scale

                    physical_values = [d * scale + offset for d in digital_values]
                    signals[ch].extend(physical_values)

        # 轉換為 numpy array
        signals = [np.array(s) for s in signals]

    return signals, signal_headers, header


# ============================================================
# 第2部分：帶通濾波
# 對應 MATLAB: eeg = bandpass(x, [1 30], fs);
# ============================================================

def bandpass_filter(signal_data, fs, low_freq=1.0, high_freq=30.0, order=4):
    """
    帶通濾波器
    對應 MATLAB 的 bandpass(x, [low high], fs)

    參數:
        signal_data: 輸入訊號
        fs: 採樣頻率 (Hz)
        low_freq: 低截止頻率 (Hz)
        high_freq: 高截止頻率 (Hz)
        order: 濾波器階數

    返回:
        filtered_signal: 濾波後的訊號
    """
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

    # 使用 filtfilt 進行零相位濾波（對應 MATLAB 的 bandpass）
    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal


# ============================================================
# 第2.5部分：重新取樣 (Resampling)
# ============================================================

def resample_signal(signal_data, fs_old, fs_new=250.0):
    """
    訊號重新取樣
    將訊號從 fs_old 轉換為 fs_new (預設 250 Hz)
    
    參數:
        signal_data: 輸入訊號 (numpy array)
        fs_old: 原始採樣率
        fs_new: 目標採樣率 (Hz)
    
    返回:
        resampled_signal: 重新取樣後的訊號
    """
    if fs_old == fs_new:
        return signal_data
    
    # 計算重新取樣後的樣本數
    num_samples = int(len(signal_data) * fs_new / fs_old)
    
    # 使用 scipy.signal.resample (傅立葉方法)
    resampled_signal = signal.resample(signal_data, num_samples)
    
    return resampled_signal


# ============================================================
# 第3部分：繪圖
# 目的: 檢測硬體擷取訊號的品質(飄移、雜訊、斷線等)，
# 因此改為分析「整段錄音」而非單一片段快照
# ============================================================

def plot_signal_quality(tt, eeg, ff, pxx, tspec, fspec, Sxx, channel_label,
                         freq_limit=30, window_start=0.0, window_duration=30.0,
                         show_fft=True, show_welch=True, save_path=None, show=True):
    """
    繪製訊號品質總覽圖：時域波形(可放大檢視任一片段)、
    原始 FFT 頻譜(對應該片段，即本專案最初的頻域圖)、
    Welch 平均頻譜(可選擇顯示與否)、時頻圖 (Spectrogram)

    參數:
        tt: 時間軸 (整段錄音)
        eeg: 濾波後的 EEG 訊號 (整段錄音)
        ff: Welch 頻譜的頻率軸
        pxx: Welch 平均功率譜密度
        tspec, fspec, Sxx: 時頻圖 (spectrogram) 的時間軸/頻率軸/功率矩陣
        channel_label: 通道名稱
        freq_limit: 頻域圖與時頻圖顯示的最高頻率 (Hz)
        window_start: 時域圖/原始 FFT 要檢視的起始時間 (秒)
        window_duration: 時域圖/原始 FFT 要檢視的長度 (秒)
        show_fft: 是否顯示所選片段的原始 FFT 頻譜子圖
        show_welch: 是否顯示 Welch 平均頻譜子圖
        save_path: 儲存路徑（None 表示不儲存）
        show: 是否顯示視窗 (預設 True)
    """
    n_rows = 1 + int(show_fft) + int(show_welch) + 1  # 時域 + FFT? + Welch? + 時頻圖
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    row = 0

    # === 1. 時域訊號 (可放大檢視整段錄音中任一片段，用於檢查斷線/削頂/突波) ===
    window_end = min(window_start + window_duration, tt[-1])
    axes[row].plot(tt, eeg, 'b-', linewidth=0.5)
    axes[row].set_xlim(window_start, window_end)
    axes[row].set_xlabel('Time (sec)')
    axes[row].set_ylabel('Amplitude')
    axes[row].set_title(f'Filtered EEG Signal ({window_start:.0f}-{window_end:.0f}s) - {channel_label}')
    axes[row].grid(True, alpha=0.3)
    row += 1

    # === 2. 原始 FFT 頻譜 (本專案最初的頻域圖，只對應時域圖選取的片段) ===
    if show_fft:
        fs = 1.0 / (tt[1] - tt[0])
        start_idx = int(round(window_start * fs))
        end_idx = int(round(window_end * fs))
        eeg_segment = eeg[start_idx:end_idx]

        ffteeg = np.abs(np.fft.fft(eeg_segment))
        T_seg = len(eeg_segment) / fs
        df_seg = 1.0 / T_seg
        n_freq_points = min(int(freq_limit / df_seg) + 1, len(ffteeg))
        ff_seg = np.arange(n_freq_points) * df_seg

        axes[row].plot(ff_seg, ffteeg[:n_freq_points], 'r-', linewidth=0.8)
        axes[row].set_xlabel('Frequency (Hz)')
        axes[row].set_ylabel('Amplitude')
        axes[row].set_title(f'FFT Spectrum ({window_start:.0f}-{window_end:.0f}s) - {channel_label}')
        axes[row].grid(True, alpha=0.3)
        row += 1

    # === 3. Welch 平均頻譜 (整段錄音的平均 PSD，比單一片段的原始 FFT 更穩定) ===
    if show_welch:
        freq_mask = ff <= freq_limit
        axes[row].semilogy(ff[freq_mask], pxx[freq_mask], 'r-', linewidth=0.8)
        axes[row].set_xlabel('Frequency (Hz)')
        axes[row].set_ylabel('PSD (Power/Hz)')
        axes[row].set_title(f'Welch Averaged Power Spectrum - {channel_label}')
        axes[row].grid(True, alpha=0.3)
        row += 1

    # === 4. 時頻圖 (檢查訊號品質是否隨時間變化) ===
    spec_freq_mask = fspec <= freq_limit
    power_db = 10 * np.log10(Sxx[spec_freq_mask, :] + 1e-12)
    im = axes[row].pcolormesh(tspec, fspec[spec_freq_mask], power_db,
                               shading='gouraud', cmap='jet')
    axes[row].set_xlabel('Time (sec)')
    axes[row].set_ylabel('Frequency (Hz)')
    axes[row].set_title(f'Spectrogram - {channel_label}')
    fig.colorbar(im, ax=axes[row], label='Power (dB)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================
# 第4部分：檔案選擇（互動式介面）
# ============================================================

def list_edf_files():
    """列出所有 EDF/BDF 檔案"""
    edf_files = []

    if not EDF_FOLDER.exists():
        print(f"錯誤: 找不到資料夾 {EDF_FOLDER}")
        return edf_files

    # 搜尋所有 .edf 和 .bdf 檔案（包含子資料夾）
    for edf_path in EDF_FOLDER.rglob("*.edf"):
        edf_files.append(edf_path)
    for bdf_path in EDF_FOLDER.rglob("*.bdf"):
        edf_files.append(bdf_path)

    return sorted(edf_files)


def display_files(edf_files):
    """顯示檔案列表"""
    print("\n" + "=" * 60)
    print("可用的 EDF/BDF 檔案")
    print("=" * 60)

    if not edf_files:
        print("沒有找到任何 EDF/BDF 檔案！")
        return

    for i, edf_path in enumerate(edf_files, 1):
        rel_path = edf_path.relative_to(EDF_FOLDER)
        file_size = edf_path.stat().st_size / 1024  # KB
        file_type = edf_path.suffix.upper()[1:]
        print(f"  [{i}] {rel_path} ({file_size:.1f} KB) [{file_type}]")

    print("=" * 60)


def select_file_interactive(edf_files):
    """互動式選擇檔案"""
    display_files(edf_files)

    if not edf_files:
        return None

    while True:
        try:
            choice = input("\n請選擇檔案編號 (輸入 q 離開): ").strip()

            if choice.lower() == 'q':
                print("已取消")
                return None

            index = int(choice) - 1
            if 0 <= index < len(edf_files):
                return edf_files[index]
            else:
                print(f"請輸入 1 到 {len(edf_files)} 之間的數字")
        except ValueError:
            print("請輸入有效的數字")


def select_time_window_interactive(total_duration, window_duration=30.0):
    """互動式選擇時域圖要放大檢視的起始時間"""
    max_start = max(0.0, total_duration - window_duration)

    while True:
        choice = input(
            f"\n請輸入時域圖要檢視的起始時間 (秒, 0~{max_start:.0f}, 預設 0): "
        ).strip()

        if choice == '':
            return 0.0

        try:
            start = float(choice)
            if 0 <= start <= max_start:
                return start
            else:
                print(f"請輸入 0 到 {max_start:.0f} 之間的數字")
        except ValueError:
            print("請輸入有效的數字")


def select_show_welch_interactive():
    """互動式選擇是否顯示 Welch 平均頻譜圖"""
    choice = input("是否顯示 Welch 平均頻譜圖? (Y/n，預設 Y): ").strip().lower()
    return choice != 'n'


def select_show_fft_interactive():
    """互動式選擇是否顯示原始 FFT 頻譜圖 (本專案最初的頻域圖)"""
    choice = input("是否顯示原始 FFT 頻譜圖? (Y/n，預設 Y): ").strip().lower()
    return choice != 'n'


def select_channel_interactive(signal_headers):
    """互動式選擇通道"""
    print("\n" + "-" * 40)
    print("可用的通道")
    print("-" * 40)

    for i, sh in enumerate(signal_headers):
        print(f"  [{i}] {sh['label']} ({sh['sample_rate']:.1f} Hz)")

    print("-" * 40)

    while True:
        try:
            choice = input("\n請選擇通道編號 (預設 0): ").strip()

            if choice == '':
                return 0

            index = int(choice)
            if 0 <= index < len(signal_headers):
                return index
            else:
                print(f"請輸入 0 到 {len(signal_headers) - 1} 之間的數字")
        except ValueError:
            print("請輸入有效的數字")


# ============================================================
# 第5部分：核心分析流程
# 對應 MATLAB 主程式邏輯
# ============================================================

def analyze_edf(edf_path, channel_index=None, window_start=None, show_fft=None, show_welch=None):
    """
    分析 EDF/BDF 檔案 - 對應 MATLAB 主程式完整流程

    MATLAB 代碼流程:
    1. edfinfo / edfread  → 讀取 EDF 檔案
    2. data.CH01          → 提取通道
    3. 計算 fs            → 取得採樣率
    4. bandpass [1,30]    → 帶通濾波
    5. fft                → 計算 FFT
    6. subplot 繪圖       → 時域 + 頻域圖

    參數:
        edf_path: EDF/BDF 檔案路徑
        channel_index: 通道索引（None 表示互動選擇）
        window_start: 時域圖要放大檢視的起始時間，秒（None 表示互動選擇）
        show_fft: 是否顯示原始 FFT 頻譜圖（None 表示互動選擇）
        show_welch: 是否顯示 Welch 平均頻譜圖（None 表示互動選擇）
    """
    edf_path = Path(edf_path)
    print(f"\n{'='*60}")
    print(f"正在讀取檔案: {edf_path.name}")
    print(f"{'='*60}")

    # === 1. 讀取 EDF 檔案 ===
    # 對應 MATLAB: info = edfinfo('...'); data = edfread('...');
    signals, signal_headers, header = read_edf(edf_path)

    # 顯示檔案資訊
    print(f"檔案格式: {header['format']} ({header['bit_depth']}-bit)")
    print(f"患者資訊: {header['patient']}")
    print(f"錄製日期: {header['startdate']} {header['starttime']}")
    print(f"通道數量: {header['n_channels']}")
    print(f"記錄時長: {header['n_records'] * header['record_duration']:.1f} 秒")

    # === 2. 選擇通道 ===
    # 對應 MATLAB: ch01 = data.CH01; x = cell2mat(ch01);
    if channel_index is None:
        channel_index = select_channel_interactive(signal_headers)

    if channel_index >= len(signals):
        print(f"警告: 通道索引 {channel_index} 超出範圍，使用通道 0")
        channel_index = 0

    x = signals[channel_index]
    channel_label = signal_headers[channel_index]['label']
    print(f"\n已提取通道: [{channel_index}] {channel_label}")

    # === 3. 計算採樣率 ===
    # 對應 MATLAB: fs = info.NumSamples(...) / seconds(info.DataRecordDuration);
    fs = signal_headers[channel_index]['sample_rate']
    print(f"採樣率: {fs} Hz")
    print(f"訊號長度: {len(x)} 點 ({len(x)/fs:.1f} 秒)")

    # === 4. 重新取樣至 250 Hz (使用者指定) ===
    # 為確保分析一致性，將訊號統一轉為 250 Hz
    fs_target = 250.0
    if fs != fs_target:
        print(f"\n正在重新取樣: {fs} Hz -> {fs_target} Hz...")
        x = resample_signal(x, fs, fs_target)
        fs = fs_target
        print(f"重新取樣完成！目前訊號點數: {len(x)}")

    # === 5. 帶通濾波 [1, 30] Hz ===
    # 對應 MATLAB: eeg = bandpass(x, [1 30], 250);
    print(f"\n正在進行帶通濾波 [1, 30] Hz...")
    eeg = bandpass_filter(x, fs, low_freq=1.0, high_freq=30.0)
    print("濾波完成！")

    # === 5. 建立時間軸 & 分析整段錄音的訊號品質 ===
    # 目的: 測試硬體訊號品質時，問題(飄移、雜訊、斷線)常是間歇性的，
    # 只看單一片段快照會漏掉，因此改為分析整段錄音
    tt = np.arange(len(x)) / fs

    # Welch 平均頻譜: 以 30 秒為一個 epoch、50% 重疊，取整段錄音的平均 PSD
    # 比單一片段的原始 FFT 更能代表整段錄音的真實頻率內容，變異也更小
    epoch_sec = 30
    nperseg = min(int(epoch_sec * fs), len(eeg))
    ff, pxx = signal.welch(eeg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # 時頻圖 (Spectrogram): 檢視訊號品質是否隨時間變化 (飄移、突波、斷線)
    spec_window_sec = 2
    spec_nperseg = min(int(spec_window_sec * fs), len(eeg))
    fspec, tspec, Sxx = signal.spectrogram(eeg, fs=fs, nperseg=spec_nperseg,
                                            noverlap=spec_nperseg // 2)

    # === 6. 選擇時域圖檢視片段 & 是否顯示 FFT / Welch 頻譜 ===
    if window_start is None:
        window_start = select_time_window_interactive(tt[-1])

    if show_fft is None:
        show_fft = select_show_fft_interactive()

    if show_welch is None:
        show_welch = select_show_welch_interactive()

    # === 7. 繪圖 ===
    print("\n正在繪製圖表...")

    output_dir = edf_path.parent
    base_name = edf_path.stem
    save_path = output_dir / f"{base_name}_analysis.png"

    plot_signal_quality(tt, eeg, ff, pxx, tspec, fspec, Sxx, channel_label,
                         freq_limit=30, window_start=window_start, show_fft=show_fft,
                         show_welch=show_welch, save_path=save_path)

    print("\n分析完成！")


# ============================================================
# 第6部分：主程式入口
# ============================================================

def main():
    """主程式 - 支援命令列參數與互動式選擇"""
    print("\n" + "=" * 60)
    print("EEG 腦波訊號分析程式")
    print("基於 MATLAB 代碼邏輯 (bandpass + FFT)")
    print("=" * 60)

    # 取得所有 EDF/BDF 檔案
    edf_files = list_edf_files()

    # 處理命令列參數
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--list' or arg == '-l':
            display_files(edf_files)
            return

        if arg == '--help' or arg == '-h':
            print(__doc__)
            return

        # 嘗試解析為檔案編號
        try:
            index = int(arg) - 1
            if 0 <= index < len(edf_files):
                analyze_edf(edf_files[index])
            else:
                print(f"錯誤: 檔案編號須在 1 到 {len(edf_files)} 之間")
        except ValueError:
            # 嘗試作為檔案路徑
            selected_file = Path(arg)
            if selected_file.exists():
                analyze_edf(selected_file)
            else:
                print(f"錯誤: 找不到檔案 {arg}")
        return

    # 互動式模式 - 支援連續分析
    while True:
        selected_file = select_file_interactive(edf_files)
        if selected_file is None:
            break

        try:
            analyze_edf(selected_file)
        except Exception as e:
            print(f"\n錯誤: {e}")
            import traceback
            traceback.print_exc()

        # 詢問是否繼續
        print("\n" + "-" * 60)
        choice = input("按 Enter 繼續分析下一個檔案，或輸入 q 離開: ").strip().lower()
        if choice == 'q':
            print("\n感謝使用！再見！")
            break

        # 重新整理檔案列表
        edf_files = list_edf_files()


if __name__ == "__main__":
    main()
