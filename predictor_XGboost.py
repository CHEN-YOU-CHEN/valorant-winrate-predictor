import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle
import argparse
import sys
import warnings
import os

warnings.filterwarnings('ignore')

# 將工作目錄切換到腳本所在的目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "scores": "maps_scores.csv",
    "overview": "overview.csv",
    "stats": "maps_stats.csv",
    "roles": "valorant_agent.csv"
}

# --- 🎯 新增：特務名稱清理函數 (用於解決 KAY/O 圖片和數據不匹配問題) ---
def clean_agent_name(name):
    """將特務名稱轉換為小寫，並移除不合法的字元 (例如 /)。"""
    if pd.isna(name): return name
    name = str(name).strip().lower()
    name = name.replace('/', '') # 移除斜線，處理 KAY/O
    name = name.replace('.', '') # 移除句點，以防 Omen.
    return name

# =====================================================================
# 階段一：資料處理與模型訓練核心函數 (加入交互特徵)
# =====================================================================

def create_and_train():
    print(">>> 正在初始化數據並訓練 XGBoost 模型...")

    # --- 1. 載入原始資料 ---
    try:
        scores = pd.read_csv(FILES["scores"], low_memory=False)
        overview = pd.read_csv(FILES["overview"], low_memory=False)
        map_stats = pd.read_csv(FILES["stats"], low_memory=False)
        agent_roles = pd.read_csv(FILES["roles"], low_memory=False)
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 {e.filename}。")
        sys.exit(1)

    # 🎯 修正點 1: 標準化特務名稱 (使用 clean_agent_name)
    agent_roles['name'] = agent_roles['name'].apply(clean_agent_name)
    role_map = agent_roles.set_index('name')['role'].str.strip().str.lower().to_dict()
    overview['Agents'] = overview['Agents'].apply(clean_agent_name) # 清洗 overview 數據

    ALL_AGENTS = sorted(list(role_map.keys()))
    match_keys = ['Tournament', 'Stage', 'Match Type', 'Match Name', 'Map']

    # 2. 建立隊伍特徵
    team_grp = overview[overview['Side'] == 'both'].groupby(
        match_keys + ['Team']
    )['Agents'].apply(list).reset_index(name='Agents_List')

    def get_full_features(agents_list):
        features = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
        for ag in agents_list:
            role = role_map.get(ag)
            if role in features:
                features[role] += 1

        for ag in ALL_AGENTS:
            features[f"agent_{ag}"] = 1 if ag in agents_list else 0

        return pd.Series(features)

    print(">>> 正在進行特徵工程...")
    feature_df = team_grp['Agents_List'].apply(get_full_features)
    team_data = pd.concat([team_grp, feature_df], axis=1)

    # 3. 建立對戰組合 (邏輯不變)
    scores['Winner_Team'] = scores.apply(
        lambda r: r['Team A'] if r['Team A Score'] > r['Team B Score'] else r['Team B'],
        axis=1
    )
    team_data = pd.merge(team_data, scores[match_keys + ['Winner_Team']], on=match_keys, how='inner')
    team_data['Win'] = (team_data['Team'] == team_data['Winner_Team']).astype(int)

    full_matchup = pd.merge(team_data, team_data, on=match_keys, suffixes=('', '_Opp'))
    full_matchup = full_matchup[full_matchup['Team'] != full_matchup['Team_Opp']]

    # 4. 地圖 winrate 清理 (邏輯不變)
    def clean_percent(x):
        try:
            return float(x.strip('%')) / 100
        except:
            return 0.5

    for c in ['Attacker Side Win Percentage', 'Defender Side Win Percentage']:
        map_stats[c] = map_stats[c].apply(clean_percent)

    map_stats_avg = map_stats.groupby('Map')[
        ['Attacker Side Win Percentage', 'Defender Side Win Percentage']
    ].mean().to_dict('index')

    full_matchup['Attacker Side Win Percentage'] = full_matchup['Map'].map(
        lambda x: map_stats_avg.get(x, {}).get('Attacker Side Win Percentage', 0.5)
    )
    full_matchup['Defender Side Win Percentage'] = full_matchup['Map'].map(
        lambda x: map_stats_avg.get(x, {}).get('Defender Side Win Percentage', 0.5)
    )

    # 5. Map One-Hot
    map_dummies = pd.get_dummies(full_matchup['Map'], prefix='Map')
    full_matchup = pd.concat([full_matchup, map_dummies], axis=1)

    agent_cols = [c for c in full_matchup.columns if c.startswith("agent_")]
    map_dummy_cols = [c for c in full_matchup.columns if c.startswith("Map_")]

    # 🎯 修正點 2: 生成 Map * Agent 交互特徵並加入 feature_cols
    print(">>> 正在生成 Map * Agent 交互特徵...")
    interaction_features = []

    for ag in agent_cols:
        for mp in map_dummy_cols:
            new_col = f"{ag}_{mp}"
            full_matchup[new_col] = full_matchup[ag] * full_matchup[mp]
            interaction_features.append(new_col)

    # 6. 建構訓練資料
    feature_template = list(feature_df.columns)

    feature_cols = (
        list(feature_df.columns)
        + [f"{c}_Opp" for c in feature_df.columns]
        + ['Attacker Side Win Percentage', 'Defender Side Win Percentage']
        + interaction_features # ✅ 交互特徵已加入
    )

    X = full_matchup[map_dummy_cols + feature_cols] # ✅ Map OHE 特徵已加入
    y = full_matchup['Win']

    # 7. 訓練模型 (使用 max_depth=8, learning_rate=0.05 等常用參數)
    print(f">>> 開始訓練 XGBoost 模型 (特徵數量: {len(X.columns)})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)

    print(f">>> 訓練集準確率: {acc_train:.2%}")
    print(f">>> 測試集準確率: {acc_test:.2%}")

    # 使用全部數據重新訓練最終模型
    model.fit(X, y)

    # 8. 儲存模型與參數
    joblib.dump(model, 'model_final.pkl')
    with open('model_columns.pkl', 'wb') as f: pickle.dump(X.columns.tolist(), f)
    with open('role_map.pkl', 'wb') as f: pickle.dump(role_map, f)
    with open('map_stats.pkl', 'wb') as f: pickle.dump(map_stats_avg, f)
    with open('meta_opponent.pkl', 'wb') as f: pickle.dump(
        full_matchup[[f"{c}_Opp" for c in feature_df.columns]].mean().to_dict(),
        f
    )
    with open('all_agents.pkl', 'wb') as f: pickle.dump(ALL_AGENTS, f)
    with open('feature_template.pkl', 'wb') as f: pickle.dump(feature_template, f)

    # 儲存交互和 OHE 輔助檔案
    with open('interaction_features.pkl', 'wb') as f: pickle.dump(interaction_features, f)
    with open('map_dummy_cols.pkl', 'wb') as f: pickle.dump(map_dummy_cols, f)
    with open('agent_cols.pkl', 'wb') as f: pickle.dump(agent_cols, f)

    print(">>> 模型與特徵儲存完成！")


# =====================================================================
# 階段二：預測 (支援 5 或 10 角色)
# =====================================================================

def predict(map_name, agents_input, model, columns, role_map, map_stats, meta_opp, all_agents, feature_template, interaction_features, map_dummy_cols, agent_cols):
    """
    預測單隊伍 (5) 或完整對戰 (10) 的勝率。
    注意：Streamlit 應用程式中呼叫此函數時，請傳遞所有 12 個參數 (包含 3 個新載入的 pkl 文件內容)。
    """
    
    # 🎯 修正點 3a: 清洗輸入名稱並切分隊伍
    cleaned_agents_input = [clean_agent_name(agent) for agent in agents_input]
    
    if len(cleaned_agents_input) == 5:
        my_agents = cleaned_agents_input
        opp_agents = None # 使用 Meta 對手
    elif len(cleaned_agents_input) == 10:
        my_agents = cleaned_agents_input[:5]
        opp_agents = cleaned_agents_input[5:] # 使用指定對手
    else:
        raise ValueError("請輸入 5 或 10 位特務。")

    # 1. 我方特徵
    my_features = {k: 0 for k in feature_template}
    counts = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}

    for agent in my_agents:
        role = role_map.get(agent)
        if role in counts:
            counts[role] += 1
        key = f"agent_{agent}"
        if key in my_features: # 使用清洗後的名稱查找特徵鍵
            my_features[key] = 1

    for r in counts:
        my_features[r] = counts[r]

    # 2. 敵方特徵
    opp_features = {}
    if opp_agents is None:
        # 使用 Meta 對手 (已從 .pkl 載入)
        for k in feature_template:
            opp_features[f"{k}_Opp"] = meta_opp[f"{k}_Opp"]
    else:
        # 計算指定敵方陣容特徵
        opp_features = {f"{k}_Opp": 0 for k in feature_template}
        counts_opp = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
        for agent in opp_agents:
            role = role_map.get(agent)
            if role in counts_opp: counts_opp[role] += 1
            key = f"agent_{agent}_Opp"
            if key in opp_features:
                opp_features[key] = 1
        for r, c in counts_opp.items(): opp_features[f"{r}_Opp"] = c

    # 3. 建立 DataFrame
    input_dict = {'Map': [map_name]}
    input_dict.update(my_features)
    input_dict.update(opp_features)
    input_dict['Attacker Side Win Percentage'] = [
        map_stats.get(map_name, {}).get('Attacker Side Win Percentage', 0.5)
    ]
    input_dict['Defender Side Win Percentage'] = [
        map_stats.get(map_name, {}).get('Defender Side Win Percentage', 0.5)
    ]

    input_df = pd.DataFrame(input_dict)
    
    # 4. Map OHE 與 交互特徵計算 (修正點 3b: 預測時計算交互特徵)
    input_encoded = pd.get_dummies(input_df, columns=['Map'])
    
    # 補齊 Map OHE 欄位
    for col in map_dummy_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # 計算交互特徵
    for col in interaction_features:
        parts = col.split('_Map_')
        agent_col = parts[0] # 例如 agent_jett
        map_col = f"Map_{parts[1]}" # 例如 Map_ascent
        
        # 確保 agent_col 存在 (因為是 Map * Agent)，如果不存在則為 0
        agent_val = input_encoded[agent_col].iloc[0] if agent_col in input_encoded.columns else 0
        map_val = input_encoded[map_col].iloc[0] if map_col in input_encoded.columns else 0
        
        input_encoded[col] = agent_val * map_val

    # 5. 對齊欄位
    final_input = input_encoded.reindex(columns=columns, fill_value=0)

    return model.predict_proba(final_input)[0][1]


# =====================================================================
# 階段三：主 CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valorant 隊伍陣容勝率預測器 (XGBoost 版)")
    parser.add_argument('map', nargs='?', help='地圖名稱 (例如: Bind)')
    parser.add_argument('agents', nargs='*', help='5 或 10 位特務名稱') # 🎯 修正點 4: 接受 5 或 10 個特務
    parser.add_argument('--retrain', action='store_true', help='重新訓練模型')
    args = parser.parse_args()

    if args.retrain or not os.path.exists('model_final.pkl'):
        create_and_train()

    if args.map and args.agents:
        # 🎯 修正點 4: 檢查輸入數量
        if len(args.agents) not in [5, 10]:
            print("請輸入 5 或 10 位特務")
            sys.exit(1)

        # 載入模型與參數 (需載入所有交互特徵相關的 .pkl 檔案)
        model = joblib.load('model_final.pkl')
        with open('model_columns.pkl', 'rb') as f: columns = pickle.load(f)
        with open('role_map.pkl', 'rb') as f: role_map = pickle.load(f)
        with open('map_stats.pkl', 'rb') as f: map_stats = pickle.load(f)
        with open('meta_opponent.pkl', 'rb') as f: meta_opp = pickle.load(f)
        with open('all_agents.pkl', 'rb') as f: all_agents = pickle.load(f)
        with open('feature_template.pkl', 'rb') as f: feature_template = pickle.load(f)
        
        # 載入交互特徵相關的輔助檔案
        with open('interaction_features.pkl', 'rb') as f: interaction_features = pickle.load(f)
        with open('map_dummy_cols.pkl', 'rb') as f: map_dummy_cols = pickle.load(f)
        with open('agent_cols.pkl', 'rb') as f: agent_cols = pickle.load(f) # 預測時沒用到，但為確保完整性可以保留
        
        # 傳遞所有 12 個參數
        rate = predict(
            args.map, args.agents, model, columns, role_map, map_stats, meta_opp, all_agents, feature_template, 
            interaction_features, map_dummy_cols, agent_cols
        )

        print("-" * 40)
        print(f"地圖: {args.map}")
        print(f"我方陣容: {', '.join(args.agents[:5])}")
        if len(args.agents) == 10:
             print(f"敵方陣容: {', '.join(args.agents[5:])}")
        print("-" * 40)
        print(f"🏆 預測勝率: {rate:.2%}")