# Valorant 陣容勝率預測器

本專案為一個基於機器學習 (Machine Learning) 的互動式 Web 應用程式，旨在透過 **Random Forest** 與 **XGBoost** 演算法，預測《特戰英豪》(Valorant) 電競比賽中特定陣容與地圖搭配的勝率。

此專案為機器學習相關課程的期末實作，展現了從 **資料清理、特徵工程、模型訓練到 Web 服務部署** 的完整機器學習開發生命週期。

## 專案動機與核心目標

在《特戰英豪》中，特務陣容的選擇（如：決鬥者、控場者、先鋒、守衛的比例）以及與地圖的契合度，對比賽勝負有著決定性的影響。本專案目標為：
1. **量化陣容效益：** 將抽象的「角色搭配」轉化為數據化的勝率預測。
2. **多維度特徵工程：** 提取並建立包含「特務角色分布」、「地圖勝率歷史數據」及「特務 × 地圖交互作用」的深度特徵。
3. **互動式決策輔助：** 開發 Streamlit 介面，讓使用者可以即時點選陣容，並直觀查看預測勝率。

## 技術棧與工具

- **資料處理與特徵工程:** `Pandas`, `NumPy`
- **機器學習模型:** `Scikit-learn` (Random Forest), `XGBoost`
- **模型儲存與載入:** `Joblib`, `Pickle`
- **前端與互動介面:** `Streamlit`, `Pillow` (PIL)

## 核心檔案結構與說明

- `app.py`: Streamlit 網頁應用程式主程式，提供直觀的視覺化 UI 介面。
- `predictor_RandomForest.py`: Random Forest 模型的資料處理、特徵工程與訓練邏輯。
- `predictor_XGboost.py`: XGBoost 模型的訓練邏輯與預測實現。
- `complexity_rf.py` / `complexity_xgb.py`: 探討模型複雜度（如決策樹深度）與預測準確率的關係，並生成超參數調優分析圖。
- `model_final.pkl`: 最終訓練完成並序列化的優化後模型。
- `accuracy_vs_complexity_plot_rf.png` / `accuracy_vs_complexity_plot_xgb.png`: 模型效能分析圖表。

## 模型成效與分析

我們比較了 Random Forest 與 XGBoost 模型，並對其進行了複雜度調優（Max Depth 測試）。
透過交叉驗證與特徵重要性分析，模型能夠有效捕捉到諸如 **"Viper 在 Breeze 地圖上的強勢度"** 等深度互動特徵。

> 詳細的超參數變化影響可參考專案內的 `accuracy_vs_complexity_plot_rf.png` 等圖表。

## 如何執行

### 步驟 1: 環境設置
請確保已安裝 Python 3.9+，接著安裝必要的套件：
```bash
pip install -r requirements.txt
```

### 步驟 2: 啟動 Streamlit 視覺化應用 (推薦)
啟動 Web 應用程式以體驗互動式介面：
```bash
streamlit run app.py
```
應用程式將自動於瀏覽器中開啟 `http://localhost:8501`。

### 步驟 3: 訓練模型與 CLI 預測 (選用)
若要重新訓練模型並生成對應的 `.pkl` 權重檔案，請於終端機執行：
```bash
python predictor_RandomForest.py --retrain
```
*(注意：重新訓練需要完整的原始資料集 `overview.csv` 等檔案，因檔案大小限制可能未包含於 Repo 中)*

**終端機直接預測測試：**
```bash
# 模式一：單隊預測 (敵方為 Meta 平均數據)
python predictor_RandomForest.py Bind jett raze viper omen killjoy

# 模式二：完整對戰預測 (10名特務)
python predictor_RandomForest.py Ascent jett raze viper omen killjoy breach neon fade astra cypher
```

## 💡 未來展望與優化方向

1. **整合即時賽事數據：** 透過 API 爬取最新賽事資料，進行 Online Learning。
2. **導入深度學習：** 嘗試使用 Neural Networks 來進一步發掘非線性特徵。
3. **更細粒度的玩家數據：** 結合玩家的 ACS (平均戰鬥分數) 與 KAST 指標，提升預測準確度。
