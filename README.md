# Trade Market Deep Learning (金融市場深度學習預測)

本專案旨在利用**深度學習 (Deep Learning)** 技術來進行金融市場的時間序列預測。提供完整的資料載入、模型訓練與結果視覺化流程，適合用於量化交易前期的策略評估與走勢預測研究。

## 專案特色 (Features)

- **資料處理**：透過 `data_loader.py` 進行金融數據的獲取與前處理。
- **模型架構對比**：
  - **基礎深度學習模型**：由 `model.py` 定義，透過 `main.py` 執行訓練。
  - **Transformer 模型**：由 `transformer_model.py` 定義，透過 `main_transformer.py` 執行訓練，利用自注意力機制 (Self-Attention) 有效捕捉時間序列的長距離依賴關係。
- **結果展示**：生成預測結果圖表，比對模型預測價格與真實市場價格的走勢差異。

## 專案架構 (Repository Structure)

```text
Trade_Market_deep_learning/
├── data_loader.py                 # 資料獲取、清洗與時間序列特徵工程
├── model.py                       # 基礎深度學習模型架構
├── main.py                        # 基礎模型的訓練與預測主程式
├── transformer_model.py           # Transformer 模型架構設計
├── main_transformer.py            # Transformer 模型的訓練與預測主程式
├── prediction_plot.png            # 基礎模型預測結果折線圖
├── prediction_plot_transformer.png# Transformer 模型預測結果折線圖
├── requirements.txt               # 專案依賴套件清單
└── .gitignore                     
```

## 1. 環境安裝 (Installation)

```bash
# 1. 複製本專案到本地端
git clone https://github.com/he451313/Trade_Market_deep_learning.git

# 2. 安裝所有必要的 Python 套件
pip install -r requirements.txt
```

## 2. 執行基礎深度學習模型 (Run Baseline Model)

執行 `main.py` 將會自動載入資料、訓練基礎模型，並在根目錄輸出 `prediction_plot.png`。

```bash
python main.py
```

## 3. 執行 Transformer 模型 (Run Transformer Model)

執行 `main_transformer.py` 將會啟動 Transformer 模型的訓練與預測，並輸出 `prediction_plot_transformer.png`。

```bash
python main_transformer.py
```
