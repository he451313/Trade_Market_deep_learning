# Trade Market Deep Learning (金融市場深度學習預測)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Time%20Series-orange)
![Transformer](https://img.shields.io/badge/Model-Transformer-success)

本專案旨在利用**深度學習 (Deep Learning)** 技術來進行金融市場的時間序列預測。專案內建了傳統深度學習模型與先進的 **Transformer 模型**，並提供完整的資料載入、模型訓練與結果視覺化流程，適合用於量化交易前期的策略評估與走勢預測研究。

## 專案特色 (Features)

- **自動化資料處理**：透過 `data_loader.py` 進行金融數據的獲取與前處理。
- **雙模型架構對比**：
  - **基礎深度學習模型**：由 `model.py` 定義，透過 `main.py` 執行訓練。
  - **Transformer 模型**：由 `transformer_model.py` 定義，透過 `main_transformer.py` 執行訓練，利用自注意力機制 (Self-Attention) 有效捕捉時間序列的長距離依賴關係。
- **直觀的結果展示**：自動生成預測結果圖表，輕鬆比對模型預測價格與真實市場價格的走勢差異。

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
└── .gitignore                     # Git 忽略清單

🚀 快速開始 (Quick Start)
1. 環境安裝 (Installation)
請確保您的環境中已安裝 Python。建議使用虛擬環境 (Virtual Environment) 來執行本專案以避免套件衝突：
code
Bash
# 1. 複製本專案到本地端
git clone https://github.com/he451313/Trade_Market_deep_learning.git
cd Trade_Market_deep_learning

# 2. 安裝所有必要的 Python 套件
pip install -r requirements.txt
2. 執行基礎深度學習模型 (Run Baseline Model)
執行 main.py 將會自動載入資料、訓練基礎模型，並在根目錄輸出 prediction_plot.png。
code
Bash
python main.py
3. 執行 Transformer 模型 (Run Transformer Model)
執行 main_transformer.py 將會啟動 Transformer 模型的訓練與預測，並輸出 prediction_plot_transformer.png。
code
Bash
python main_transformer.py
📊 成果展示 (Results)
本專案將預測結果視覺化，以下是兩種不同神經網路架構在金融時間序列上的預測表現比對：
📈 基礎模型預測 (Baseline Model)
執行 main.py 所產生的預測價格與實際價格對比。
![alt text](https://raw.githubusercontent.com/he451313/Trade_Market_deep_learning/master/prediction_plot.png)
📈 Transformer 模型預測 (Transformer Model)
執行 main_transformer.py 所產生的預測價格與實際價格對比。
![alt text](https://raw.githubusercontent.com/he451313/Trade_Market_deep_learning/master/prediction_plot_transformer.png)
