import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path
import os

# 將工作目錄切換到腳本所在的目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 導入模型和函數 ---
try:
    from predictor_RandomForest import predict
    import joblib, pickle
except ImportError as e:
    st.error(f"❌ 模組導入錯誤。請確保 'predictor_RandomForest.py' 存在。錯誤詳情: {e}")
    st.stop()

# ------------------------ 載入模型和參數 ------------------------
try:
    model = joblib.load('model_final.pkl')
    with open('model_columns.pkl','rb') as f: columns = pickle.load(f)
    with open('role_map.pkl','rb') as f: role_map = pickle.load(f)
    with open('map_stats.pkl','rb') as f: map_stats = pickle.load(f)
    with open('meta_opponent.pkl','rb') as f: meta_opp = pickle.load(f)
    with open('feature_template.pkl','rb') as f: feature_template = pickle.load(f)
    with open('interaction_features.pkl','rb') as f: interaction_features = pickle.load(f)
    with open('map_dummy_cols.pkl','rb') as f: map_dummy_cols = pickle.load(f)

    ALL_AGENTS = list(role_map.keys())
except Exception as e:
    st.error(f"❌ 模型檔案載入失敗。請確認所有 .pkl 檔案存在。\n錯誤：{e}")
    st.stop()


# ------------------------ UI 設定 (圖片等寬修正) ------------------------
st.set_page_config(
    page_title="Valorant 勝率預測器",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>

.block-container { padding-top: 2rem; }

/* 卡片容器 */
.agent-card-container {
    padding: 8px;
    border-radius: 10px;
    background-color: #1e1e1e;
    text-align: center;
    margin-bottom: 15px;
    border: 3px solid transparent;
    transition: all 0.2s ease-in-out;
    display: flex;
    flex-direction: column;
    width: 100%;
}

/* 選中狀態紅框 */
.selected-agent {
    border: 3px solid #ff4655;
    box-shadow: 0 0 10px #ff4655;
}

/* 讓 st.image 完全佔滿容器 */
div.stImage {
    display: grid !important;
    place-items: center !important;
    width: 100%;
}

/* 🎯 圖片寬度 = 按鈕寬度（保持比例） */
.stImage img {
    width: 100% !important;
    height: auto !important;   /* ⭐保持長寬比 */
    display: block;
    border-radius: 6px;
}

/* 分類標題 */
.section-title {
    color: #ff4655;
    font-size: 26px;
    font-weight: bold;
    padding: 10px 0;
    border-bottom: 2px solid #333;
    margin-bottom: 10px;
}

/* 按鈕樣式 */
button {
    background-color: #ff4655 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)


MAPS = ['Bind','Haven','Ascent','Split','Icebox','Breeze','Fracture','Abyss','Lotus','Pearl']
AGENT_IMAGES = {agent: f"images/agents/{agent}.png" for agent in ALL_AGENTS}


# ------------------------ 頁面標題 ------------------------
st.markdown("<h1 style='text-align:center;color:white;'>🎯 Valorant 陣容勝率預測器</h1>", unsafe_allow_html=True)
st.markdown("---")
map_name = st.selectbox("🗺 選擇地圖", MAPS)


# ------------------------ 初始化狀態 ------------------------
if 'my_agents' not in st.session_state: st.session_state['my_agents'] = []
if 'opp_agents' not in st.session_state: st.session_state['opp_agents'] = []


# ------------------------ 切換特務 ------------------------
def toggle_agent(agent, selected_list_key, limit=5):
    selected = st.session_state[selected_list_key]
    if agent in selected:
        selected.remove(agent)
    elif len(selected) < limit:
        selected.append(agent)
    st.session_state[selected_list_key] = selected


# ------------------------ 🔥 圖片不變形的卡片按鈕 ------------------------
def agent_card_button(agent, is_selected, selected_list_key, limit=5):
    img_path = Path(AGENT_IMAGES.get(agent, ""))

    if img_path.exists():
        img = Image.open(img_path)
        if not is_selected:
            img = ImageOps.grayscale(img)
    else:
        img = Image.new("RGB", (300, 300), color=(80,80,80))

    card_class = "agent-card-container selected-agent" if is_selected else "agent-card-container"
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

    # 🎯 會自動保持長寬比
    st.image(img, use_container_width=True)

    st.button(
        f"{agent.capitalize()} {'✔' if is_selected else ''}",
        key=f"btn_{selected_list_key}_{agent}",
        on_click=toggle_agent,
        args=(agent, selected_list_key, limit),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------ 固定 5 欄的特務展示 ------------------------
def render_agent_selection_fixed(agent_list, session_key, title, limit):
    st.markdown(f'<div class="section-title">🛡 {title} ({limit} 名)</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, agent in enumerate(agent_list):
        col_index = i % 5
        with cols[col_index]:
            agent_card_button(agent, agent in st.session_state[session_key], session_key, limit)


# ------------------------ 我方選角 ------------------------
render_agent_selection_fixed(ALL_AGENTS, "my_agents", "我方陣容", 5)
st.info(f"✔ **我方已選擇 ({len(st.session_state['my_agents'])}/5):** {', '.join(a.capitalize() for a in st.session_state['my_agents'])}")


# ------------------------ 敵方選角 ------------------------
render_agent_selection_fixed(ALL_AGENTS, "opp_agents", "敵方陣容 (未選則使用平均)", 5)
st.info(f"✔ **敵方已選擇 ({len(st.session_state['opp_agents'])}/5):** {', '.join(a.capitalize() for a in st.session_state['opp_agents'])}")


# ------------------------ 預測按鈕 ------------------------
st.markdown("---")
predict_btn = st.button("🏆 開始預測 🎯", use_container_width=True)

if predict_btn:
    if len(st.session_state['my_agents']) != 5:
        st.error("❌ 請選擇完整的我方 5 名陣容！")
    else:
        agents_input = st.session_state['my_agents'].copy()
        if len(st.session_state['opp_agents']) == 5:
            agents_input += st.session_state['opp_agents']

        rate = predict(
            map_name,
            agents_input,
            model,
            columns,
            role_map,
            map_stats,
            meta_opp,
            ALL_AGENTS,
            feature_template,
            map_dummy_cols,
            interaction_features
        )

        st.markdown("## 🏅 預測結果", unsafe_allow_html=True)
        st.progress(int(rate * 100))

        st.markdown(f"""
        <h1 style='color:#ff4655; font-size: 72px; text-align:center;'>
            {rate*100:.2f}%
        </h1>
        """, unsafe_allow_html=True)
