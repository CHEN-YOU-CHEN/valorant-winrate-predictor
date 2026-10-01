import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle
import argparse
import sys
import warnings
import os

warnings.filterwarnings('ignore')

# 將工作目錄切換到腳本所在的目錄，確保能正確讀取相對路徑的檔案
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "scores": "maps_scores.csv",
    "overview": "overview.csv",
    "stats": "maps_stats.csv",
    "roles": "valorant_agent.csv"
}

# =====================================================================
# 階段一：資料處理與模型訓練核心函數 (加入交互特徵)
# =====================================================================

def create_and_train():
    """執行數據載入、特徵工程、模型訓練與儲存的完整流程。"""
    print(">>> 正在初始化數據並訓練 Random Forest 模型...")
    
    # --- 1. 載入原始資料 ---
    try:
        scores = pd.read_csv(FILES["scores"], low_memory=False)
        overview = pd.read_csv(FILES["overview"], low_memory=False)
        map_stats = pd.read_csv(FILES["stats"], low_memory=False)
        agent_roles = pd.read_csv(FILES["roles"], low_memory=False)
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 {e.filename}。請檢查檔案名稱是否正確。")
        sys.exit(1)

    # 準備特務基礎資料 (邏輯不變)
    agent_roles['name'] = agent_roles['name'].str.strip().str.lower()
    role_map = agent_roles.set_index('name')['role'].str.strip().str.lower().to_dict()
    overview['Agents'] = overview['Agents'].str.strip().str.lower()
    ALL_AGENTS = sorted(list(role_map.keys()))
    match_keys = ['Tournament', 'Stage', 'Match Type', 'Match Name', 'Map']
    
    # 2. 建立隊伍特徵
    team_grp = overview[overview['Side'] == 'both'].groupby(match_keys + ['Team'])['Agents'].apply(list).reset_index(name='Agents_List')

    # 特徵工程函數：計算角色數量和具體特務選角 (邏輯不變)
    def get_full_features(agents_list):
        features = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
        for agent in agents_list:
            role = role_map.get(agent)
            if role in features: features[role] += 1
        
        for agent_name in ALL_AGENTS:
            features[f"agent_{agent_name}"] = 1 if agent_name in agents_list else 0
            
        return pd.Series(features)

    print(">>> 正在進行特徵工程...")
    feature_df = team_grp['Agents_List'].apply(get_full_features)
    team_data = pd.concat([team_grp, feature_df], axis=1)

    # 3. 建立對戰組合 (我方 vs 對方)
    def get_winner(row):
        return row['Team A'] if row['Team A Score'] > row['Team B Score'] else row['Team B']
    scores['Winner_Team'] = scores.apply(get_winner, axis=1)
    
    team_data = pd.merge(team_data, scores[match_keys + ['Winner_Team']], on=match_keys, how='inner')
    team_data['Win'] = (team_data['Team'] == team_data['Winner_Team']).astype(int)

    # 自連結：將同一場比賽的對手陣容合併進來 (產生我方 vs 敵方特徵)
    full_matchup = pd.merge(team_data, team_data, on=match_keys, suffixes=('', '_Opp'))
    full_matchup = full_matchup[full_matchup['Team'] != full_matchup['Team_Opp']]

    # 4. 加入地圖數據 (攻守勝率，邏輯不變)
    def clean_percent(x):
        try: return float(x.strip('%')) / 100 if isinstance(x, str) else x
        except: return 0.5
    for c in ['Attacker Side Win Percentage', 'Defender Side Win Percentage']:
        map_stats[c] = map_stats[c].apply(clean_percent)
    
    map_stats_avg = map_stats.groupby('Map')[['Attacker Side Win Percentage', 'Defender Side Win Percentage']].mean().to_dict('index')

    full_matchup['Attacker Side Win Percentage'] = full_matchup['Map'].map(lambda x: map_stats_avg.get(x, {}).get('Attacker Side Win Percentage', 0.5))
    full_matchup['Defender Side Win Percentage'] = full_matchup['Map'].map(lambda x: map_stats_avg.get(x, {}).get('Defender Side Win Percentage', 0.5))

    # --- 5. 核心升級：生成 Map * Agent 交互作用特徵 ---
    
    # 將地圖 One-Hot 展開
    map_dummies = pd.get_dummies(full_matchup['Map'], prefix='Map')
    full_matchup = pd.concat([full_matchup, map_dummies], axis=1)

    # 識別所有具體特務欄位 (我方和敵方)
    agent_cols = [c for c in full_matchup.columns if c.startswith('agent_')]
    map_dummy_cols = [c for c in full_matchup.columns if c.startswith('Map_')]
    
    print(">>> 正在生成 Map * Agent 交互特徵...")
    interaction_features = []
    
    # 遍歷所有特務和所有地圖的組合
    for agent_col in agent_cols:
        for map_col in map_dummy_cols:
            # 創建新特徵: Agent_Jett * Map_Bind
            new_col_name = f"{agent_col}_{map_col}"
            full_matchup[new_col_name] = full_matchup[agent_col] * full_matchup[map_col]
            interaction_features.append(new_col_name)

    # 6. 模型訓練準備
    # 組合所有特徵欄位
    feature_cols = [c for c in feature_df.columns] + \
                   [f"{c}_Opp" for c in feature_df.columns] + \
                   ['Attacker Side Win Percentage', 'Defender Side Win Percentage'] + \
                   interaction_features # <<< 加入交互特徵

    X = full_matchup[map_dummy_cols + feature_cols]
    y = full_matchup['Win']
    
    # 儲存特徵模板和欄位 (確保預測時能正確對齊所有交互特徵)
    feature_template_updated = list(feature_df.columns)
    
    # --- 7. 訓練與評估 (使用最佳參數) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f">>> 開始訓練 Random Forest 模型 (特徵總數: {len(X.columns)})...")
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8, 
        min_samples_leaf=10, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 評估準確度
    test_accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)

    print(f">>> 模型訓練完成！訓練集準確率 (Fit): {train_accuracy:.2%}")
    print(f"🎉 測試集準確率 (Generalization): {test_accuracy:.2%}")
    
    # 最終模型：使用全部數據重新訓練一次
    model.fit(X, y) 

    # 8. 儲存所有必要檔案 (儲存最新的交互特徵欄位)
    meta_opponent = full_matchup[[f"{c}_Opp" for c in feature_df.columns]].mean().to_dict()
    joblib.dump(model, 'model_final.pkl')
    with open('model_columns.pkl', 'wb') as f: pickle.dump(X.columns.tolist(), f) # 儲存包含交互特徵的完整欄位
    with open('map_stats.pkl', 'wb') as f: pickle.dump(map_stats_avg, f)
    with open('meta_opponent.pkl', 'wb') as f: pickle.dump(meta_opponent, f)
    with open('feature_template.pkl', 'wb') as f: pickle.dump(list(feature_df.columns), f)
    
    # 為了讓 predict 函數使用，我們需要儲存互動特徵名稱
    with open('interaction_features.pkl', 'wb') as f: pickle.dump(interaction_features, f)
    with open('map_dummy_cols.pkl', 'wb') as f: pickle.dump(map_dummy_cols, f)
    with open('agent_cols.pkl', 'wb') as f: pickle.dump(agent_cols, f)
    
    print(">>> 模型與參數儲存完成。")
# ... (略去 predict 函數，因為修改複雜度高，需專注在訓練)

# =====================================================================
# 階段二：預測功能
# =====================================================================

def predict(map_name, agents, model, columns, role_map, map_stats, meta_opp, all_agents, feature_template):
    """預測單一隊伍陣容在某地圖上，對陣 Meta 對手的勝率。"""
    
    # 1. 建構我方特徵 (與訓練時的欄位名稱和順序保持一致)
    my_features = {k: 0 for k in feature_template}
    counts = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
    
    for agent in agents:
        clean_agent = agent.lower().strip()
        role = role_map.get(clean_agent)
        
        if role in counts: counts[role] += 1
        if f"agent_{clean_agent}" in my_features:
            my_features[f"agent_{clean_agent}"] = 1
            
    for r, c in counts.items():
        my_features[r] = c
        
    # 2. 建構輸入 DataFrame
    input_dict = {'Map': [map_name]}
    
    # 填充我方特徵和敵方 Meta 特徵
    for k, v in my_features.items():
        input_dict[k] = [v]
    for k in feature_template:
        input_dict[f"{k}_Opp"] = [meta_opp[f"{k}_Opp"]]
        
    # 填充地圖數據
    input_dict['Attacker Side Win Percentage'] = [map_stats.get(map_name, {}).get('Attacker Side Win Percentage', 0.5)]
    input_dict['Defender Side Win Percentage'] = [map_stats.get(map_name, {}).get('Defender Side Win Percentage', 0.5)]
    
    input_df = pd.DataFrame(input_dict)
    
    # 3. 對齊格式並預測
    input_encoded = pd.get_dummies(input_df, columns=['Map'])
    input_final = input_encoded.reindex(columns=columns, fill_value=0)
    
    # 輸出預測勝率 (模型預測為 1 (Win) 的機率)
    return model.predict_proba(input_final)[0][1]

# =====================================================================
# 階段三：程式主執行區塊 (CLI 介面)
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valorant 隊伍陣容勝率預測器")
    parser.add_argument('map', nargs='?', help='地圖名稱 (例如: Bind)')
    parser.add_argument('agents', nargs='*', help='5 位特務名稱 (例如: Jett Raze Viper Omen Killjoy)')
    parser.add_argument('--retrain', action='store_true', help='強制重新訓練模型')

    args = parser.parse_args()

    # 檢查是否需要重新訓練
    if args.retrain or not os.path.exists('model_final.pkl'):
        create_and_train()

    if args.map and args.agents:
        if len(args.agents) != 5:
            print("[錯誤] 請輸入 5 位特務。")
            sys.exit(1)
            
        # 載入所有儲存的參數
        try:
            model = joblib.load('model_final.pkl')
            with open('model_columns.pkl', 'rb') as f: columns = pickle.load(f)
            with open('role_map.pkl', 'rb') as f: role_map = pickle.load(f)
            with open('map_stats.pkl', 'rb') as f: map_stats = pickle.load(f)
            with open('meta_opponent.pkl', 'rb') as f: meta_opp = pickle.load(f)
            with open('all_agents.pkl', 'rb') as f: all_agents = pickle.load(f)
            with open('feature_template.pkl', 'rb') as f: feature_template = pickle.load(f)
        except Exception as e:
            print(f"檔案讀取錯誤，請確認所有 .pkl 檔案存在：{e}")
            sys.exit(1)
            
        rate = predict(args.map, args.agents, model, columns, role_map, map_stats, meta_opp, all_agents, feature_template)
        
        # 簡化輸出
        print("-" * 40)
        print(f"地圖: {args.map}")
        print(f"我方陣容: {', '.join(args.agents)}")
        print("-" * 40)
        print(f"🏆 預測勝率: {rate:.2%}")