https://eeg-analysis-tool-niehfhtem8valk9rd2nuse.streamlit.app/
# EEG 腦波訊號判讀小程式

這是一個用於分析腦波訊號 (EEG) 的 Python 應用程式。支援讀取 EDF/BDF 格式檔案，進行訊號預處理（帶通濾波）、頻譜分析 (FFT)，並繪製時域與頻域的視覺化圖表。

本專案提供 **網頁版 (Streamlit)** 與 **桌面版 (Windows .exe)** 兩種使用方式，方便不同平台的使用者操作。

## ✨ 主要功能

- **檔案支援**: 支援讀取標準 EDF (16-bit) 與 BioSemi BDF (24-bit) 格式。
- **訊號處理**: 內建 Butterworth 帶通濾波器 (預設 1-30 Hz)，有效去除雜訊。
- **頻譜分析**: 使用快速傅立葉變換 (FFT) 分析訊號頻率成分。
- **視覺化**: 自動繪製濾波後的時域波形圖與 FFT 頻譜圖。
- **互動介面**:
  - **網頁版**: 上傳檔案、選擇通道、調整濾波參數、下載分析圖表。
  - **桌面版**: 自動搜尋檔案、互動式選擇通道。

## 🚀 快速開始

### 方式一：使用網頁版 (推薦 macOS/iOS/Windows)

無需安裝 Python，直接開啟網頁瀏覽器即可使用。

1. 確保已安裝 Python 依賴套件：
   ```bash
   pip install -r requirements.txt
   ```
2. 啟動網頁應用程式：
   ```bash
   streamlit run app.py
   ```
3. 瀏覽器會自動開啟，直接拖放 `.edf` 檔案即可開始分析。

### 方式二：使用 Windows 執行檔 (.exe)

無需安裝 Python環境。

1. 下載 `EEG_Analysis_Tool.exe`。
2. 將 `.edf` 資料夾放在執行檔的同一目錄下。
3. 點擊執行檔即可開始分析。

### 方式三：直接執行 Python 腳本

```bash
# 執行主程式 (命令行互動模式)
python main.py

# 列出所有可用檔案
python main.py --list

# 直接分析特定檔案
python main.py 1
```

## 📂 檔案結構

- **`app.py`**: Streamlit 網頁應用程式主程式。
- **`main.py`**: 核心分析邏輯與命令行介面程式。
- **`requirements.txt`**: 專案依賴套件清單。
- **`README.md`**: 專案說明文件。

## 🛠️ 技術棧

- **Python 3.x**
- **NumPy & SciPy**: 數值計算與訊號處理。
- **Matplotlib**: 繪圖與數據視覺化。
- **Streamlit**: 網頁應用程式框架。

## 📝 注意事項

- 本程式預設濾波範圍為 1-30 Hz，可在網頁版中手動調整。
- 桌面版執行時，請確保 `.edf` 檔案路徑正確或位於程式同一目錄下的 `.edf` 資料夾內。

---
*Created by Wayne*
