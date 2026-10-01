import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os

# 將工作目錄切換到腳本所在的目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 設置檔案名稱 (使用程式碼中定義的檔案)
FILES = {
    "scores": "maps_scores.csv",
    "overview": "overview.csv",
    "stats": "maps_stats.csv",
    "roles": "valorant_agent.csv"
}

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# ============== 完整特徵工程邏輯 (從 create_and_train 複製) ==============
def prepare_data_for_plot():
    """執行完整的特徵工程，輸出 X 和 Y。"""
    try:
        scores = pd.read_csv(FILES["scores"], low_memory=False)
        overview = pd.read_csv(FILES["overview"], low_memory=False)
        map_stats = pd.read_csv(FILES["stats"], low_memory=False)
        agent_roles = pd.read_csv(FILES["roles"], low_memory=False)
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 {e.filename}。請確認所有 CSV 檔案存在。")
        sys.exit(1)

    agent_roles['name'] = agent_roles['name'].str.strip().str.lower()
    role_map = agent_roles.set_index('name')['role'].str.strip().str.lower().to_dict()
    overview['Agents'] = overview['Agents'].str.strip().str.lower()
    ALL_AGENTS = sorted(list(role_map.keys()))
    match_keys = ['Tournament', 'Stage', 'Match Type', 'Match Name', 'Map']
    
    # 2. 建立隊伍特徵
    team_grp = overview[overview['Side'] == 'both'].groupby(match_keys + ['Team'])['Agents'].apply(list).reset_index(name='Agents_List')

    def get_full_features(agents_list):
        features = {'duelist': 0, 'controller': 0, 'initiator': 0, 'sentinel': 0}
        for agent in agents_list:
            role = role_map.get(agent)
            if role in features: features[role] += 1
        for agent_name in ALL_AGENTS:
            features[f"agent_{agent_name}"] = 1 if agent_name in agents_list else 0
        return pd.Series(features)

    feature_df = team_grp['Agents_List'].apply(get_full_features)
    team_data = pd.concat([team_grp, feature_df], axis=1)

    # 3. 建立對戰組合 (我方 vs 對方)
    def get_winner(row):
        return row['Team A'] if row['Team A Score'] > row['Team B Score'] else row['Team B']
    scores['Winner_Team'] = scores.apply(get_winner, axis=1)
    
    team_data = pd.merge(team_data, scores[match_keys + ['Winner_Team']], on=match_keys, how='inner')
    team_data['Win'] = (team_data['Team'] == team_data['Winner_Team']).astype(int)

    full_matchup = pd.merge(team_data, team_data, on=match_keys, suffixes=('', '_Opp'))
    full_matchup = full_matchup[full_matchup['Team'] != full_matchup['Team_Opp']]

    # 4. 加入地圖數據 (攻守勝率)
    def clean_percent(x):
        try: return float(x.strip('%')) / 100 if isinstance(x, str) else x
        except: return 0.5
    for c in ['Attacker Side Win Percentage', 'Defender Side Win Percentage']:
        map_stats[c] = map_stats[c].apply(clean_percent)
    
    map_stats_avg = map_stats.groupby('Map')[['Attacker Side Win Percentage', 'Defender Side Win Percentage']].mean().to_dict('index')

    full_matchup['Attacker Side Win Percentage'] = full_matchup['Map'].map(lambda x: map_stats_avg.get(x, {}).get('Attacker Side Win Percentage', 0.5))
    full_matchup['Defender Side Win Percentage'] = full_matchup['Map'].map(lambda x: map_stats_avg.get(x, {}).get('Defender Side Win Percentage', 0.5))

    # --- 5. 核心升級：生成 Map * Agent 交互作用特徵 ---
    map_dummies = pd.get_dummies(full_matchup['Map'], prefix='Map')
    full_matchup = pd.concat([full_matchup, map_dummies], axis=1)
    agent_cols = [c for c in full_matchup.columns if c.startswith('agent_')]
    map_dummy_cols = [c for c in full_matchup.columns if c.startswith('Map_')]
    
    print(">>> 正在生成 Map * Agent 交互特徵...")
    interaction_features = []
    for agent_col in agent_cols:
        for map_col in map_dummy_cols:
            new_col_name = f"{agent_col}_{map_col}"
            full_matchup[new_col_name] = full_matchup[agent_col] * full_matchup[map_col]
            interaction_features.append(new_col_name)

    # 6. 模型訓練準備
    feature_cols = [c for c in feature_df.columns] + \
                   [f"{c}_Opp" for c in feature_df.columns] + \
                   ['Attacker Side Win Percentage', 'Defender Side Win Percentage'] + \
                   interaction_features

    X = full_matchup[map_dummies.columns.tolist() + feature_cols]
    y = full_matchup['Win']
    
    return X, y

# ======================================================================

# 1. 準備數據
print(">>> 載入並準備數據...")
X_full, y_full = prepare_data_for_plot()
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# 2. 定義要測試的最大深度範圍
max_depths = [2, 4, 6, 8, 10, 12, 15, 20] 

train_scores = []
test_scores = []

# 3. 循環訓練模型並記錄準確率
print(">>> 開始運行 max_depth 複雜度測試...")
for depth in max_depths:
    print(f"   - 訓練 Depth = {depth}...")
    # 保持 n_estimators 和 min_samples_leaf 不變，只改變 max_depth
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=depth, 
        min_samples_leaf=10, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)

# 4. 繪製並儲存圖表
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, label='Training Set Accuracy', marker='o', color='blue')
plt.plot(max_depths, test_scores, label='Test Set Accuracy', marker='o', color='red')

# 標記您選擇的最佳點 (max_depth=8)
optimal_depth_index = max_depths.index(8)
plt.axvline(x=8, color='green', linestyle='--', linewidth=1.5, label='Optimal Depth (Max Depth = 8)')
plt.scatter(8, test_scores[optimal_depth_index], color='green', s=100, zorder=5)

plt.title('Accuracy vs. Model Complexity (Max Depth)')
plt.xlabel('Max Depth of Trees ')
plt.ylabel('Accuracy ')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()

# 儲存圖表
output_filename = 'accuracy_vs_complexity_plot_rf.png'
plt.savefig(output_filename)
print(f"✅ 準確率曲線圖表已儲存為 {output_filename}")