import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle
import argparse
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# 將工作目錄切換到腳本所在的目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "scores": "maps_scores.csv",
    "overview": "overview.csv",
    "stats": "maps_stats.csv",
    "roles": "valorant_agent.csv"
}

# =====================================================================
# 階段一：資料處理與模型訓練核心函數
# =====================================================================

def clean_agent_name(name):
    """將特務名稱轉換為小寫，並移除非法字元，確保與檔案名稱兼容 (e.g., KAY/O -> kayo)。"""
    if pd.isna(name): return name
    name = str(name).strip().lower()
    name = name.replace('/', '') # 移除斜線，處理 KAY/O
    name = name.replace('.', '') # 移除句點，以防 Omen.
    return name

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
        print(f"錯誤：找不到檔案 {e.filename}。")
        sys.exit(1)

    # 準備特務基礎資料
    agent_roles['name'] = agent_roles['name'].apply(clean_agent_name)
    role_map = agent_roles.set_index('name')['role'].str.strip().str.lower().to_dict()
    overview['Agents'] = overview['Agents'].apply(clean_agent_name)
    ALL_AGENTS = sorted(list(role_map.keys()))
    match_keys = ['Tournament', 'Stage', 'Match Type', 'Match Name', 'Map']

    # 2. 建立隊伍特徵
    team_grp = overview[overview['Side'] == 'both'].groupby(match_keys + ['Team'])['Agents'].apply(list).reset_index(name='Agents_List')

    # 特徵工程函數
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

    # 3. 建立對戰組合
    scores['Winner_Team'] = scores.apply(lambda row: row['Team A'] if row['Team A Score'] > row['Team B Score'] else row['Team B'], axis=1)
    team_data = pd.merge(team_data, scores[match_keys + ['Winner_Team']], on=match_keys, how='inner')
    team_data['Win'] = (team_data['Team'] == team_data['Winner_Team']).astype(int)

    full_matchup = pd.merge(team_data, team_data, on=match_keys, suffixes=('', '_Opp'))
    full_matchup = full_matchup[full_matchup['Team'] != full_matchup['Team_Opp']]

    # 4. 加入地圖數據
    def clean_percent(x):
        try: return float(x.strip('%')) / 100 if isinstance(x, str) else x
        except: return 0.5

    for c in ['Attacker Side Win Percentage', 'Defender Side Win Percentage']:
        map_stats[c] = map_stats[c].apply(clean_percent)
    map_stats_avg = map_stats.groupby('Map')[['Attacker Side Win Percentage', 'Defender Side Win Percentage']].mean().to_dict('index')

    full_matchup['Attacker Side Win Percentage'] = full_matchup['Map'].map(lambda x: map_stats_avg.get(x, {}).get('Attacker Side Win Percentage', 0.5))
    full_matchup['Defender Side Win Percentage'] = full_matchup['Map'].map(lambda x: map_stats_avg.get(x, {}).get('Defender Side Win Percentage', 0.5))

    # 5. Map One-Hot
    map_dummies = pd.get_dummies(full_matchup['Map'], prefix='Map')
    full_matchup = pd.concat([full_matchup, map_dummies], axis=1)

    # 6. Map × Agent 交互特徵
    agent_cols = [c for c in full_matchup.columns if c.startswith('agent_')]
    map_dummy_cols = [c for c in full_matchup.columns if c.startswith('Map_')]
    print(">>> 生成 Map * Agent 交互特徵...")
    interaction_features = []
    for agent_col in agent_cols:
        for map_col in map_dummy_cols:
            new_col_name = f"{agent_col}_{map_col}"
            full_matchup[new_col_name] = full_matchup[agent_col] * full_matchup[map_col]
            interaction_features.append(new_col_name)

    # 7. 訓練準備
    feature_cols = [c for c in feature_df.columns] + \
                   [f"{c}_Opp" for c in feature_df.columns] + \
                   ['Attacker Side Win Percentage', 'Defender Side Win Percentage'] + \
                   interaction_features

    X = full_matchup[map_dummy_cols + feature_cols]
    y = full_matchup['Win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f">>> 訓練 Random Forest 模型 (特徵總數: {len(X.columns)})...")
    model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f">>> 訓練集準確率: {model.score(X_train, y_train):.2%}")
    print(f">>> 測試集準確率: {model.score(X_test, y_test):.2%}")

    # 全部數據重訓
    model.fit(X, y)

    # 儲存
    meta_opponent = full_matchup[[f"{c}_Opp" for c in feature_df.columns]].mean().to_dict()
    joblib.dump(model, 'model_final.pkl')
    with open('model_columns.pkl', 'wb') as f: pickle.dump(X.columns.tolist(), f)
    with open('map_stats.pkl', 'wb') as f: pickle.dump(map_stats_avg, f)
    with open('meta_opponent.pkl', 'wb') as f: pickle.dump(meta_opponent, f)
    with open('feature_template.pkl', 'wb') as f: pickle.dump(list(feature_df.columns), f)
    with open('interaction_features.pkl', 'wb') as f: pickle.dump(interaction_features, f)
    with open('map_dummy_cols.pkl', 'wb') as f: pickle.dump(map_dummy_cols, f)
    with open('agent_cols.pkl', 'wb') as f: pickle.dump(agent_cols, f)

    # 儲存 role_map 與 all_agents
    with open('role_map.pkl', 'wb') as f: pickle.dump(role_map, f)
    with open('all_agents.pkl', 'wb') as f: pickle.dump(ALL_AGENTS, f)
    print(">>> 模型與特徵已儲存完成。")

# =====================================================================
# 階段二：預測功能
# =====================================================================

def predict(map_name, agents_input, model, columns, role_map, map_stats, meta_opp, all_agents, feature_template, map_dummy_cols, interaction_features):
    """支援 5 或 10 名特務預測"""
    
    # 修正 1：先清洗原始輸入列表
    cleaned_agents_input = [clean_agent_name(agent) for agent in agents_input] 

    if len(cleaned_agents_input) == 5:
        my_agents = cleaned_agents_input
        opp_agents = None
    elif len(cleaned_agents_input) == 10:
        my_agents = cleaned_agents_input[:5]
        opp_agents = cleaned_agents_input[5:]
    else:
        raise ValueError("請輸入 5 或 10 位特務。")

    
    # 我方特徵
    my_features = {k: 0 for k in feature_template}
    counts = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
    
    # 修正 2：直接遍歷 my_agents (它現在包含清洗後的名稱)
    for agent in my_agents: 
        # agent 變數已經是 clean_agent_name 處理過的結果
        role = role_map.get(agent) 
        
        if role in counts: counts[role] += 1
        if f"agent_{agent}" in my_features: my_features[f"agent_{agent}"] = 1
        
    for r, c in counts.items(): my_features[r] = c

    # 敵方特徵 (同步修正)
    opp_features = {}
    if opp_agents is None:
        for k in feature_template: opp_features[f"{k}_Opp"] = meta_opp[f"{k}_Opp"]
    else:
        opp_features = {f"{k}_Opp": 0 for k in feature_template}
        counts_opp = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
        
        # 修正 3：直接遍歷 opp_agents (它現在包含清洗後的名稱)
        for agent in opp_agents:
            role = role_map.get(agent)
            if role in counts_opp: counts_opp[role] += 1
            if f"agent_{agent}_Opp" in opp_features: opp_features[f"agent_{agent}_Opp"] = 1
            
        for r, c in counts_opp.items(): opp_features[f"{r}_Opp"] = c

    # ... (其餘邏輯，如建立 DataFrame, One-Hot Map, 交互特徵, 對齊欄位，保持不變)
    
    # 建立 DataFrame
    input_dict = {'Map':[map_name]}
    input_dict.update(my_features)
    input_dict.update(opp_features)
    input_dict['Attacker Side Win Percentage'] = [map_stats.get(map_name, {}).get('Attacker Side Win Percentage',0.5)]
    input_dict['Defender Side Win Percentage'] = [map_stats.get(map_name, {}).get('Defender Side Win Percentage',0.5)]
    input_df = pd.DataFrame(input_dict)

    # One-Hot Map
    input_encoded = pd.get_dummies(input_df, columns=['Map'])
    for col in map_dummy_cols:
        if col not in input_encoded: input_encoded[col] = 0

    # 交互特徵
    for col in interaction_features:
        parts = col.split('_Map_')
        agent_col = parts[0]
        map_col = f"Map_{parts[1]}"
        val = 0
        if agent_col in input_encoded.columns and map_col in input_encoded.columns:
            val = input_encoded[agent_col] * input_encoded[map_col]
        input_encoded[col] = val

    # 對齊欄位
    input_final = input_encoded.reindex(columns=columns, fill_value=0)
    return model.predict_proba(input_final)[0][1]

# =====================================================================
# 主程式
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valorant 隊伍勝率預測器")
    parser.add_argument('map', nargs='?', help='地圖名稱')
    parser.add_argument('agents', nargs='*', help='5 或 10 位特務')
    parser.add_argument('--retrain', action='store_true', help='重新訓練模型')
    args = parser.parse_args()

    if args.retrain or not os.path.exists('model_final.pkl'):
        create_and_train()

    if args.map and args.agents:
        if len(args.agents) not in [5,10]:
            print("[錯誤] 請輸入 5 或 10 位特務。")
            sys.exit(1)

        # 載入模型與參數
        model = joblib.load('model_final.pkl')
        with open('model_columns.pkl','rb') as f: columns = pickle.load(f)
        with open('role_map.pkl','rb') as f: role_map = pickle.load(f)
        with open('map_stats.pkl','rb') as f: map_stats = pickle.load(f)
        with open('meta_opponent.pkl','rb') as f: meta_opp = pickle.load(f)
        with open('all_agents.pkl','rb') as f: all_agents = pickle.load(f)
        with open('feature_template.pkl','rb') as f: feature_template = pickle.load(f)
        with open('interaction_features.pkl','rb') as f: interaction_features = pickle.load(f)
        with open('map_dummy_cols.pkl','rb') as f: map_dummy_cols = pickle.load(f)

        rate = predict(args.map, args.agents, model, columns, role_map, map_stats, meta_opp, all_agents, feature_template, map_dummy_cols, interaction_features)

        print("-"*40)
        print(f"地圖: {args.map}")
        print(f"我方陣容: {', '.join(args.agents[:5])}")
        if len(args.agents)==10: print(f"敵方陣容: {', '.join(args.agents[5:])}")
        print("-"*40)
        print(f"🏆 預測勝率: {rate:.2%}")
