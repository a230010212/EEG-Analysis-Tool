"""
EEG 腦波訊號分析網頁版 (Streamlit)
"""
import streamlit as st
import numpy as np
import tempfile
import os
from pathlib import Path
from scipy import signal
import main as backend  # 匯入 main.py 作為後端邏輯

st.set_page_config(page_title="EEG 腦波訊號分析", layout="wide")

st.title("EEG 腦波訊號分析")
st.markdown("上傳 EDF/BDF 檔案進行時頻域分析")

# 檔案上傳
uploaded_file = st.file_uploader("選擇 EDF/BDF 檔案", type=["edf", "bdf"])

if uploaded_file is not None:
    # 儲存暫存檔 (因為 backend.read_edf 需要檔案路徑)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        st.divider()
        st.subheader("1. 檔案資訊")
        
        # 讀取檔案
        with st.spinner("正在讀取檔案..."):
            signals, signal_headers, header = backend.read_edf(tmp_path)
            
        # 顯示標頭資訊
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"檔案格式: {header['format']} ({header['bit_depth']}-bit)")
            st.text(f"患者資訊: {header['patient']}")
        with col2:
            st.text(f"錄製日期: {header['startdate']} {header['starttime']}")
            st.text(f"記錄時長: {header['n_records'] * header['record_duration']:.1f} 秒")
            st.text(f"通道數量: {header['n_channels']}")

        st.divider()
        st.subheader("2. 選擇通道")
        
        # 建立通道選項列表
        channel_options = [f"[{i}] {sh['label']} ({sh['sample_rate']:.1f} Hz)" 
                          for i, sh in enumerate(signal_headers)]
        
        selected_option = st.selectbox("請選擇要分析的通道", channel_options)
        
        # 解析選擇的索引
        channel_index = int(selected_option.split(']')[0].strip('['))
        
        # 計算錄音總長度，供時域檢視片段選擇使用
        # 注意: signal_headers[...]['n_samples'] 是 EDF 規格中「每個 data record
        # 的取樣點數」，不是全檔案總點數，因此改用實際訊號陣列長度計算
        total_duration = len(signals[channel_index]) / signal_headers[channel_index]['sample_rate']

        # 參數設定
        with st.expander("進階設定"):
            low_freq = st.number_input("低截止頻率 (Hz)", value=1.0, min_value=0.1)
            high_freq = st.number_input("高截止頻率 (Hz)", value=30.0, min_value=1.0)
            show_welch = st.checkbox("顯示 Welch 平均頻譜", value=True)

            if total_duration > 1:
                default_end = min(30.0, total_duration)
                window_start, window_end = st.slider(
                    "時域圖檢視片段 (秒)，可拖曳兩端調整長度、拖曳整段移動位置",
                    min_value=0.0, max_value=float(total_duration),
                    value=(0.0, default_end), step=1.0,
                )
                window_duration = window_end - window_start
            else:
                window_start = 0.0
                window_duration = total_duration
                st.caption(f"錄音長度僅 {total_duration:.1f} 秒，時域圖將顯示整段錄音")

        if st.button("開始分析", type="primary"):
            st.divider()
            st.subheader("3. 分析結果")
            
            # 提取數據
            x = signals[channel_index]
            fs = signal_headers[channel_index]['sample_rate']
            label = signal_headers[channel_index]['label']
            
            # 重新取樣至 250 Hz (使用者指定)
            fs_target = 250.0
            if fs != fs_target:
                x = backend.resample_signal(x, fs, fs_target)
                fs = fs_target

            # 帶通濾波 (整段訊號)
            with st.spinner("正在進行訊號處理..."):
                eeg = backend.bandpass_filter(x, fs, low_freq=low_freq, high_freq=high_freq)

                tt = np.arange(len(eeg)) / fs

                # Welch 平均頻譜: 以 30 秒為一個 epoch、50% 重疊，取整段錄音的平均 PSD
                # 用於測試硬體訊號品質時，比單一片段的原始 FFT 更能代表整段錄音
                nperseg = min(int(30 * fs), len(eeg))
                ff, pxx = signal.welch(eeg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

                # 時頻圖 (Spectrogram): 檢視訊號品質是否隨時間變化 (飄移、突波、斷線)
                spec_nperseg = min(int(2 * fs), len(eeg))
                fspec, tspec, Sxx = signal.spectrogram(eeg, fs=fs, nperseg=spec_nperseg,
                                                        noverlap=spec_nperseg // 2)

                # 繪圖（顯示範圍隨高截止頻率調整，但不超過 Nyquist）
                nyquist = fs / 2
                freq_limit = min(high_freq, nyquist)
                fig = backend.plot_signal_quality(tt, eeg, ff, pxx, tspec, fspec, Sxx,
                                                   label, freq_limit=freq_limit,
                                                   window_start=window_start,
                                                   window_duration=window_duration,
                                                   show_welch=show_welch, show=False)
                
                # 顯示圖表
                st.pyplot(fig)
                
                # 建立下載按鈕
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                file_name = f"{Path(uploaded_file.name).stem}_{label}_analysis.png"
                
                st.download_button(
                    label="📥 下載分析圖片",
                    data=buf,
                    file_name=file_name,
                    mime="image/png"
                )
                
                st.success("分析完成！")

    except Exception as e:
        st.error(f"發生錯誤: {e}")
    finally:
        # 清理暫存檔
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
