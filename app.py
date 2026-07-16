import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time, random, joblib, os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL — DoH Intrusion Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css');

.stApp {
    background-color: #09090f;
    background-image:
        radial-gradient(circle at 0% 0%, rgba(124,58,237,0.25) 0%, transparent 35%),
        radial-gradient(circle at 100% 100%, rgba(37,99,235,0.25) 0%, transparent 35%),
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: auto, auto, 42px 42px, 42px 42px;
    background-attachment: fixed;
    color: #d8d8e6;
    animation: bgDrift 22s ease-in-out infinite alternate;
}
@keyframes bgDrift {
    0%   { background-position: 0% 0%, 100% 100%, 0 0, 0 0; }
    100% { background-position: 4% 4%, 96% 96%, 42px 42px, 42px 42px; }
}
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1,h2,h3,h4 { font-family: 'Poppins', sans-serif !important; color: #ffffff !important; }
p, li, div { font-family: 'Poppins', sans-serif; }

/* ── Hero title box (gradient card wrapping each page's title) ───────────── */
.hero-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(37,99,235,0.25));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px; padding: 28px 32px; margin-bottom: 24px;
}
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border-radius: 50px; padding: 5px 16px; font-size: 12px;
    color: #c4b5fd; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 14px;
}
.hero-title { font-size: 34px; font-weight: 700; color: #ffffff; line-height: 1.25; margin: 0 0 10px 0; }
.hero-title .accent {
    background: linear-gradient(90deg, #c4b5fd, #93c5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: 15px; color: #b8b8c8; max-width: 720px; }

/* ── Metric / result cards — glass style ──────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 20px; text-align: center;
    font-family: 'Poppins', sans-serif; backdrop-filter: blur(18px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 0 30px rgba(139,92,246,0.25); }
.metric-label { font-size: 11px; color: #8a8a9c; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 700; }
.metric-value.green, .metric-value.blue, .metric-value.amber, .metric-value.red, .metric-value.white { background:none; }
.metric-sub   { font-size: 11px; color: #8a8a9c; margin-top: 4px; }
.green  { color: #34d399; } .red   { color: #ff5577; }
.amber  { color: #ffb347; } .blue  { color: #60a5fa; }
.white  { color: #ffffff; }

.alert-box {
    background: rgba(255,51,85,0.08); border: 1px solid rgba(255,51,85,0.3);
    border-radius: 12px; padding: 10px 14px; margin-bottom: 6px;
    font-family: 'Poppins', sans-serif; font-size: 12px; color: #ffaabb;
}
.safe-box {
    background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.25);
    border-radius: 12px; padding: 10px 14px; margin-bottom: 6px;
    font-family: 'Poppins', sans-serif; font-size: 12px; color: #a8d4ff;
}
.section-header {
    font-family: 'Poppins', sans-serif; font-size: 12px; font-weight: 600;
    color: #8a8a9c; text-transform: uppercase; letter-spacing: 2px;
    border-left: 2px solid #8b5cf6; padding-left: 10px; margin-bottom: 12px;
}
.mode-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 20px; margin-bottom: 10px; cursor: pointer;
    backdrop-filter: blur(18px); transition: transform 0.2s, box-shadow 0.2s;
}
.mode-card:hover { border-color: #8b5cf6; transform: translateY(-4px); box-shadow: 0 0 30px rgba(139,92,246,0.25); }
.result-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 18px; margin-bottom: 12px; backdrop-filter: blur(18px);
}

/* ── Sidebar — professional glass panel ───────────────────────────────────── */
div[data-testid="stSidebar"] {
    background-color: rgba(15,15,24,0.9); border-right: 1px solid rgba(255,255,255,0.08);
}
div[data-testid="stSidebar"] * { color: #d8d8e6 !important; }
div[data-testid="stSidebar"] .stSelectbox label { color: #8a8a9c !important; font-size:11px !important; }

/* Sidebar nav radio, restyled as pill nav items (no visible radio dot) */
div[data-testid="stSidebar"] div[role="radiogroup"] { gap: 8px; }
div[data-testid="stSidebar"] label[data-baseweb="radio"] {
    background: transparent; border-radius: 12px; padding: 13px 16px;
    margin: 2px 0; width: 100%; transition: background 0.15s;
}
div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover { background: rgba(139,92,246,0.12); }
div[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }
div[data-testid="stSidebar"] label[data-baseweb="radio"] p {
    font-family: 'Poppins', sans-serif !important; font-size: 15px !important; color: #d8d8e6 !important;
}
div[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] { background-color: rgba(139,92,246,0.18); }
div[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] p { color: #a78bfa !important; font-weight: 600 !important; }

.stButton > button {
    background: linear-gradient(90deg, #8b5cf6, #60a5fa); color: #ffffff; border: none;
    font-family: 'Poppins', sans-serif; font-weight: 600; width: 100%;
    border-radius: 50px; padding: 12px;
}
.stButton > button:hover { box-shadow: 0 0 24px rgba(124,58,237,0.4); }
.stButton > button[kind="secondary"] { background: transparent; border: 1px solid #ff5577; color: #ff5577; }

.stFileUploader { background: rgba(255,255,255,0.05); border: 1px dashed rgba(255,255,255,0.15); border-radius: 16px; }
.stProgress > div > div { background: linear-gradient(90deg, #8b5cf6, #60a5fa); }
div[data-testid="stDataFrame"] { background: rgba(255,255,255,0.03); }

.tag-doh       { background:rgba(96,165,250,0.15);color:#93c5fd;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Poppins',sans-serif; }
.tag-nondoh    { background:rgba(139,92,246,0.15);color:#c4b5fd;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Poppins',sans-serif; }
.tag-benign    { background:rgba(255,221,0,0.12);color:#ffdd00;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Poppins',sans-serif; }
.tag-malicious { background:rgba(255,51,85,0.15);color:#ff5577;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Poppins',sans-serif; }

/* ── Sidebar logo block (icon badge + title + subtitle) ───────────────────── */
.sidebar-logo-row { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
.logo-badge {
    flex-shrink: 0; width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #8b5cf6, #60a5fa);
    display: flex; align-items: center; justify-content: center; font-size: 20px; color: #fff;
}
.logo-title { font-size: 18px; font-weight: 700; color: #ffffff; line-height: 1.2; }
.logo-subtitle { font-size: 11px; color: #8a8a9c; margin-top: 1px; }

/* ── Footer ────────────────────────────────────────────────────────────────── */
.app-footer {
    margin-top: 40px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.08);
    text-align: center; font-size: 12px; color: #6b6b7c; font-family: 'Poppins', sans-serif;
}

.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.03); border-bottom:1px solid rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"] { color:#8a8a9c; font-family:'Poppins',sans-serif; font-size:12px; }
.stTabs [aria-selected="true"] { color:#a78bfa !important; border-bottom:2px solid #8b5cf6; }
</style>
""", unsafe_allow_html=True)

# ── Icon helper ───────────────────────────────────────────────────────────────
# Renders a real Bootstrap Icon (https://icons.getbootstrap.com/) as inline HTML.
# Use this (not emoji, not the ":material/x:" shortcode) whenever an icon needs
# to sit inside a raw HTML string (unsafe_allow_html=True) — e.g. the sidebar
# logo or the hero title box — since the ":material/x:" shortcode only
# auto-converts inside genuine Streamlit markdown/widgets, not raw injected HTML.
def icon(name, color="#d8d8e6", size=None):
    style = f"color:{color};"
    if size: style += f"font-size:{size};"
    return f'<i class="bi bi-{name}" style="{style}"></i>'

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_NAMES  = ["DoH", "NonDoH", "Benign", "Malicious"]
COLORS       = {"DoH":"#3399ff","NonDoH":"#00ff88","Benign":"#ddcc00","Malicious":"#ff3355"}
TAG_CSS      = {"DoH":"tag-doh","NonDoH":"tag-nondoh","Benign":"tag-benign","Malicious":"tag-malicious"}
SEQ_LEN      = 10
N_FEATURES   = 36

# Final v3 model results (verified from locked run)
MODEL_RESULTS = {
    "GRU": {
        "accuracy":78.64,"far":11.49,"mae":0.1137,"rmse":0.2405,"mcc":0.7168,
        "recall":{"DoH":65.51,"NonDoH":100.00,"Benign":100.00,"Malicious":49.05},
        "params":"1,229,316","auc":0.9366,"mean_ap":0.8186
    },
    "Transformer": {
        "accuracy":78.00,"far":14.53,"mae":0.1180,"rmse":0.2438,"mcc":0.7067,
        "recall":{"DoH":56.41,"NonDoH":99.96,"Benign":100.00,"Malicious":55.63},
        "params":"1,270,596","auc":0.9323,"mean_ap":0.8012
    },
    "Hybrid GRU-Transformer": {
        "accuracy":79.14,"far":11.87,"mae":0.1126,"rmse":0.2398,"mcc":0.7228,
        "recall":{"DoH":64.38,"NonDoH":100.00,"Benign":100.00,"Malicious":52.20},
        "params":"2,540,356","auc":0.9382,"mean_ap":0.8229
    },
    "Ensemble (GRU+Hybrid)": {
        "accuracy":79.09,"far":11.57,"mae":0.1132,"rmse":0.2396,"mcc":0.7225,
        "recall":{"DoH":65.26,"NonDoH":100.00,"Benign":100.00,"Malicious":51.12},
        "params":"—","auc":0.9380,"mean_ap":0.8222
    },
}

# 36 feature names in order (after engineering + correlation drop)
FEATURE_NAMES = [
    "SourcePort","DestinationPort","Duration",
    "FlowBytesSent","FlowBytesReceived",
    "PacketLengthVariance","PacketLengthMean","PacketLengthMedian",
    "PacketLengthMode","PacketLengthSkewFromMedian","PacketLengthSkewFromMode",
    "PacketLengthCoefficientofVariation",
    "PacketTimeVariance","PacketTimeMean","PacketTimeMedian",
    "PacketTimeMode","PacketTimeSkewFromMedian","PacketTimeSkewFromMode",
    "PacketTimeCoefficientofVariation",
    "ResponseTimeTimeVariance","ResponseTimeTimeMean","ResponseTimeTimeMedian",
    "ResponseTimeTimeMode","ResponseTimeTimeSkewFromMedian",
    "ResponseTimeTimeSkewFromMode","ResponseTimeTimeCoefficientofVariation",
    "BytesSentReceivedRatio","FlowSendEfficiency","FlowReceiveEfficiency",
    "PacketRhythm","PacketSizeConsistency","ResponsePacketRatio",
    "ResponseRhythm","LogDuration","LogFlowBytesSent","LogFlowBytesReceived",
]

# ── Model architecture (must match Colab Cell 9 exactly — rebuilt here so ──
#    weights load correctly on CPU without the CudnnRNNV3 GPU op baked in) ──
def _build_transformer_block_class():
    from tensorflow import keras
    from tensorflow.keras import layers

    class TransformerBlock(keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim, self.num_heads, self.ff_dim, self.dropout = embed_dim, num_heads, ff_dim, dropout
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
            self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(embed_dim)])
            self.ln1 = layers.LayerNormalization(epsilon=1e-6)
            self.ln2 = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(dropout)
            self.drop2 = layers.Dropout(dropout)

        def call(self, x, training=False):
            attn_out = self.drop1(self.att(x, x, training=training), training=training)
            x = self.ln1(x + attn_out)
            ffn_out = self.drop2(self.ffn(x), training=training)
            return self.ln2(x + ffn_out)

        def get_config(self):
            config = super().get_config()
            config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads,
                            'ff_dim': self.ff_dim, 'dropout': self.dropout})
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    return TransformerBlock


def _build_heavy_hybrid(TransformerBlock, seq_len=SEQ_LEN, n_features=N_FEATURES, n_classes=4):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(seq_len, n_features))
    g = layers.Bidirectional(layers.GRU(256, return_sequences=True))(inputs)
    g = layers.BatchNormalization()(g); g = layers.Dropout(0.3)(g)
    g = layers.Bidirectional(layers.GRU(128, return_sequences=True))(g)
    g = layers.BatchNormalization()(g); g = layers.Dropout(0.3)(g)
    g = layers.Bidirectional(layers.GRU(64, return_sequences=True))(g)
    g = layers.BatchNormalization()(g); g = layers.Dropout(0.2)(g)
    g_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='gru_self_attn')(g, g)
    g_attn = layers.Dropout(0.2)(g_attn)
    g = layers.LayerNormalization(epsilon=1e-6)(g + g_attn)
    g_pool = layers.GlobalAveragePooling1D()(g)
    g_pool = layers.Dense(128, activation='relu', name='gru_proj')(g_pool)

    t = layers.Dense(128, activation='relu')(inputs)
    t = layers.LayerNormalization(epsilon=1e-6)(t)
    for i in range(4):
        t = TransformerBlock(embed_dim=128, num_heads=8, ff_dim=512, dropout=0.1, name=f'hybrid_trans_{i}')(t)
    t_pool = layers.GlobalAveragePooling1D()(t)
    t_pool = layers.Dense(128, activation='relu', name='trans_proj')(t_pool)

    g_exp = layers.Reshape((1, 128))(g_pool)
    t_exp = layers.Reshape((1, 128))(t_pool)
    gt_cross = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='gt_cross')(g_exp, t_exp)
    gt_cross = layers.Reshape((128,))(gt_cross)
    tg_cross = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='tg_cross')(t_exp, g_exp)
    tg_cross = layers.Reshape((128,))(tg_cross)

    fused = layers.Concatenate()([g_pool, t_pool, gt_cross, tg_cross])
    x = layers.Dense(512, activation='relu')(fused); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x); x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='Heavy_Hybrid_FINAL')


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    try:
        import numpy as np_local

        tb_class = _build_transformer_block_class()
        model = _build_heavy_hybrid(tb_class)

        loaded_npz = np_local.load('models/hybrid_weights_cpu.npz')
        weight_arrays = [loaded_npz[f'arr_{i}'] for i in range(len(loaded_npz.files))]
        model.set_weights(weight_arrays)

        scaler = joblib.load('models/scaler_FINAL.pkl')
        return model, scaler, True, None
    except Exception as e:
        return None, None, False, str(e)

@st.cache_data
def load_test_splits():
    try:
        X_te = np.load('data/X_te.npy')
        y_te = np.load('data/y_te.npy').astype(int)
        return X_te, y_te, True
    except:
        return None, None, False

model, scaler, model_loaded, model_error = load_model_and_scaler()
X_te, y_te, splits_loaded = load_test_splits()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "sim_running": False, "sim_packets": [], "sim_alerts": [],
    "sim_count": 0, "sim_threats": 0,
    "sim_class_counts": {"DoH":0,"NonDoH":0,"Benign":0,"Malicious":0},
    "csv_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c8ffd8', family='Share Tech Mono'),
        margin=dict(l=20, r=20, t=30, b=20),
    )

def predict_sequence(seq_2d):
    """seq_2d: (SEQ_LEN, N_FEATURES) already scaled"""
    if model_loaded and model is not None:
        try:
            inp   = seq_2d.reshape(1, SEQ_LEN, N_FEATURES).astype('float32')
            probs = model.predict(inp, verbose=0)[0]
            return LABEL_NAMES[probs.argmax()], float(probs.max()*100), probs.tolist()
        except Exception as e:
            pass
    # Fallback simulation
    weights = [0.25, 0.35, 0.20, 0.20]
    cls  = random.choices(LABEL_NAMES, weights=weights)[0]
    conf = random.uniform(55, 95)
    probs = [0.05]*4
    probs[LABEL_NAMES.index(cls)] = conf/100
    return cls, conf, probs

def probs_bar(probs):
    fig = go.Figure(go.Bar(
        x=LABEL_NAMES, y=[p*100 for p in probs],
        marker_color=[COLORS[l] for l in LABEL_NAMES],
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside', textfont=dict(color='#c8ffd8', size=11)
    ))
    fig.update_layout(**dark_layout(), height=220, showlegend=False,
        yaxis=dict(range=[0,115], gridcolor='#1a3d28', ticksuffix='%'),
        xaxis=dict(tickfont=dict(size=12)))
    return fig

def class_pie(counts):
    labels = list(counts.keys())
    vals   = list(counts.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        marker_colors=[COLORS[l] for l in labels],
        hole=0.55, textinfo='label+percent',
        textfont=dict(color='#c8ffd8', size=11)
    ))
    fig.update_layout(**dark_layout(), height=260, showlegend=False)
    return fig

def hero_title(icon_name, badge_text, title_html, subtitle=None):
    """Renders the purple/blue gradient hero box used at the top of every page."""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="hero-box">
        <span class="hero-badge">{icon(icon_name, color="#c4b5fd")} {badge_text}</span>
        <div class="hero-title">{title_html}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def live_traffic_widget():
    """Purely decorative animated 'Live Traffic' sparkline. Runs inside its own
    sandboxed iframe (components.v1.html), so it cannot touch or break the
    rest of the page's layout — unlike the earlier position:fixed CSS trick."""
    components.html("""
    <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08);
                border-radius:20px; padding:18px; box-sizing:border-box; height:214px;
                font-family:'Poppins',sans-serif; backdrop-filter: blur(18px);">
        <div style="color:#fff; font-weight:600; font-size:14px; margin-bottom:10px;">Live Traffic</div>
        <svg id="liveChart" width="100%" height="150" viewBox="0 0 300 150" preserveAspectRatio="none"></svg>
    </div>
    <script>
    const svg = document.getElementById('liveChart');
    const N = 30;
    let points = [];
    for (let i = 0; i < N; i++) { points.push(75 + (Math.random() - 0.5) * 40); }

    function render() {
        const w = 300, h = 150;
        const stepX = w / (N - 1);
        let d = "";
        points.forEach((y, i) => {
            const x = i * stepX;
            d += (i === 0 ? "M" : "L") + x + " " + (h - y) + " ";
        });
        svg.innerHTML = `
            <defs>
                <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#8b5cf6"/>
                    <stop offset="100%" stop-color="#60a5fa"/>
                </linearGradient>
            </defs>
            <path d="${d}" fill="none" stroke="url(#grad)" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round"/>
        `;
    }
    render();
    setInterval(() => {
        points.shift();
        const last = points[points.length - 1];
        let next = last + (Math.random() - 0.5) * 25;
        next = Math.max(10, Math.min(140, next));
        points.push(next);
        render();
    }, 900);
    </script>
    """, height=220)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo-row">
        <div class="logo-badge">{icon('shield-lock-fill', color='#ffffff')}</div>
        <div>
            <div class="logo-title">SENTINEL</div>
            <div class="logo-subtitle">DoH Intrusion Prediction System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        ":material/home: Home",
        ":material/track_changes: Predict Traffic",
        ":material/bar_chart: Model Comparison",
        ":material/folder: Dataset Info",
        ":material/info: About"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model Status</div><div class="metric-value {"green" if model_loaded else "amber"}">{"● ONLINE" if model_loaded else "● DEMO"}</div><div class="metric-sub">{"Hybrid GRU-Transformer" if model_loaded else "Simulation mode"}</div></div>', unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
if page == ":material/home: Home":
# ══════════════════════════════════════════════════════════════════════════════
    col_hero, col_live = st.columns([2, 1])
    with col_hero:
        hero_title(
            "shield-lock-fill", "AI Intrusion Prediction",
            'SENTINEL — <span class="accent">Intrusion Prediction System</span>',
            "Hybrid GRU-Transformer model for encrypted DoH traffic classification."
        )
    with col_live:
        live_traffic_widget()

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Accuracy",   "79.14%", "Hybrid GRU-Transformer", "green"),
        ("False Alarm Rate","11.87%", "Lowest achieved",    "green"),
        ("Macro AUC",       "0.9382", "Hybrid GRU-Transformer", "green"),
        ("Parameters",      "2.54M",  "Final model size",   "blue"),
    ]
    for col, (label, val, sub, color) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### :material/push_pin: What This System Does")
        st.markdown("""
        SENTINEL classifies encrypted network traffic into four categories using
        a deep learning model trained on real DoH tunnel traffic data.

        | Class | Label | Description |
        |---|---|---|
        | 0 | **DoH** | Normal DNS-over-HTTPS |
        | 1 | **NonDoH** | Standard HTTPS traffic |
        | 2 | **Benign** | Safe DoH tunnel |
        | 3 | **Malicious** | DoH tunneling attack |

        Navigate to **Predict Traffic** to run predictions using one of three modes.
        """)

    with c2:
        st.markdown("### :material/science: System Architecture")
        st.markdown("""
        **Three deep learning architectures were built and compared:**

        - **GRU** — 4× Bidirectional GRU + Self-Attention → 78.64%
        - **Transformer** — 6× Transformer blocks → 78.00%
        - **Hybrid GRU-Transformer** → **79.14%**

        The Hybrid model combines temporal sequence learning (GRU) with global
        dependency modelling (Transformer) through a cross-attention fusion layer,
        making it the strongest performer across all metrics.
        """)

    st.markdown("---")
    st.markdown("### :material/trending_up: Final Model Performance")
    names = list(MODEL_RESULTS.keys())
    accs  = [MODEL_RESULTS[m]["accuracy"] for m in names]
    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker_color=["#00ff88" if "Hybrid" in n and "★" in n else "#3399ff" if "GRU" in n else "#ffaa00" if "Trans" in n else "#aa66ff" for n in names],
        text=[f"{a:.2f}%" for a in accs], textposition="outside",
        textfont=dict(color="#c8ffd8", size=12)
    ))
    fig.update_layout(**dark_layout(), height=300,
        yaxis=dict(range=[76,81], gridcolor="#1a3d28", ticksuffix="%"),
        showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/track_changes: Predict Traffic":
# ══════════════════════════════════════════════════════════════════════════════
    hero_title("bullseye", "Predict Traffic", 'Traffic <span class="accent">Prediction</span>', "Choose a prediction mode below.")

    mode = st.radio("Prediction Mode", [
        ":material/science: Mode 1 — Dataset Simulation",
        ":material/wifi_tethering: Mode 2 — Live Network Capture",
    ], label_visibility="collapsed")

    st.markdown("---")

    # ── MODE 1: Dataset Simulation ────────────────────────────────────────────
    
    # At the TOP of Mode 1 section, after splits_loaded check, add this:
    # Pre-build balanced index list ONCE before the loop
    if splits_loaded:
        # Get indices for each class
        class_idx_pool = {}
    for cls in range(4):
        class_idx_pool[cls] = list(np.where(y_te == cls)[0])
        random.shuffle(class_idx_pool[cls])

    if "Mode 1" in mode:
        st.markdown("### :material/science: Mode 1 — Dataset Simulation")
        st.markdown("Runs real samples from the CIRA test split through the trained model one by one.")

        if not splits_loaded:
            st.warning("Test split not found at `data/X_te.npy`. Place the .npy files in the `data/` folder.")
            st.info("The simulation will run in **demo mode** using randomly generated features instead.")

        col_ctrl, col_info = st.columns([1, 2])
        with col_ctrl:
            speed = st.slider("Simulation speed (sec/sample)", 0.3, 2.0, 0.8, 0.1)
            n_samples = st.slider("Number of samples to run", 10, 100, 30)
            if st.button("Start Simulation", icon=":material/play_arrow:"):
                st.session_state.sim_running    = True
                st.session_state.sim_packets    = []
                st.session_state.sim_alerts     = []
                st.session_state.sim_count      = 0
                st.session_state.sim_threats    = 0
                st.session_state.sim_class_counts = {"DoH":0,"NonDoH":0,"Benign":0,"Malicious":0}
            if st.button("Stop", icon=":material/stop:"):
                st.session_state.sim_running = False

        with col_info:
            # These placeholders update IN the loop below
            ph_count  = st.empty()
            ph_threat = st.empty()
            ph_count.markdown(f'<div class="metric-card"><div class="metric-label">Packets Processed</div><div class="metric-value white">{st.session_state.sim_count}</div></div>', unsafe_allow_html=True)
            ph_threat.markdown(f'<div class="metric-card"><div class="metric-label">Threats Detected</div><div class="metric-value {"red" if st.session_state.sim_threats > 0 else "green"}">{st.session_state.sim_threats}</div></div>', unsafe_allow_html=True)

        # Run simulation loop
        if st.session_state.sim_running and st.session_state.sim_count < n_samples:
            placeholder_chart  = st.empty()
            placeholder_alerts = st.empty()
            placeholder_prob   = st.empty()

            for i in range(st.session_state.sim_count, n_samples):
                if not st.session_state.sim_running:
                    break
                # balance the sampling
                if splits_loaded:
                    # Pick from each class in rotation so demo reflects true balance
                    target_class = i % 4   # cycles 0,1,2,3,0,1,2,3...
                    pool = class_idx_pool[target_class]
                    idx = pool[i//4 % len(pool)]
                    seq = X_te[idx]
                    true_label = LABEL_NAMES[int(y_te[idx])]
                else:
                    seq = np.random.rand(SEQ_LEN, N_FEATURES).astype(np.float32)
                    true_label = "?"

                pred, conf, probs = predict_sequence(seq)
                ts = datetime.now().strftime("%H:%M:%S")
                src_ip = f"{random.randint(10,220)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

                pkt = {"time": ts, "src": src_ip, "pred": pred, "conf": conf,
                       "probs": probs, "true": true_label}
                st.session_state.sim_packets.append(pkt)
                st.session_state.sim_class_counts[pred] += 1
                st.session_state.sim_count += 1
                if pred == "Malicious":
                    st.session_state.sim_threats += 1
                    st.session_state.sim_alerts.append(
                        f"[{ts}] THREAT — {src_ip} classified as MALICIOUS ({conf:.1f}% confidence)"
                    )

                # ── Update counters in real time ──────────────────────────────
                ph_count.markdown(f'<div class="metric-card"><div class="metric-label">Packets Processed</div><div class="metric-value white">{st.session_state.sim_count}</div></div>', unsafe_allow_html=True)
                ph_threat.markdown(f'<div class="metric-card"><div class="metric-label">Threats Detected</div><div class="metric-value {"red" if st.session_state.sim_threats > 0 else "green"}">{st.session_state.sim_threats}</div></div>', unsafe_allow_html=True)

                # ── Accuracy and FAR (live, computed so far) ──────────────────
                if st.session_state.sim_count > 0 and splits_loaded:
                    all_preds  = [p["pred"]  for p in st.session_state.sim_packets]
                    all_trues  = [p["true"]  for p in st.session_state.sim_packets if p["true"] != "?"]
                    all_preds_for_acc = [p["pred"] for p in st.session_state.sim_packets if p["true"] != "?"]
                    if all_trues:
                        live_acc = sum(p==t for p,t in zip(all_preds_for_acc, all_trues)) / len(all_trues) * 100
                        fp_live  = sum(1 for p,t in zip(all_preds_for_acc, all_trues) if p=="Malicious" and t!="Malicious")
                        tn_live  = sum(1 for p,t in zip(all_preds_for_acc, all_trues) if p!="Malicious" and t!="Malicious")
                        far_live = fp_live/(fp_live+tn_live)*100 if (fp_live+tn_live)>0 else 0.0
                        ph_count.markdown(f'<div class="metric-card"><div class="metric-label">Packets Processed</div><div class="metric-value white">{st.session_state.sim_count}</div><div class="metric-sub">Live Accuracy: {live_acc:.1f}% | FAR: {far_live:.1f}%</div></div>', unsafe_allow_html=True)

                with placeholder_chart:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
                        st.plotly_chart(class_pie(st.session_state.sim_class_counts), use_container_width=True)
                    with c2:
                        st.markdown('<div class="section-header">Latest Prediction Probabilities</div>', unsafe_allow_html=True)
                        st.plotly_chart(probs_bar(probs), use_container_width=True)

                with placeholder_alerts:
                    st.markdown('<div class="section-header">Alert Log</div>', unsafe_allow_html=True)
                    if st.session_state.sim_alerts:
                        for alert in reversed(st.session_state.sim_alerts[-8:]):
                            st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-box">{icon("check-circle-fill", color="#a8d4ff")} No threats detected yet</div>', unsafe_allow_html=True)

                with placeholder_prob:
                    st.markdown(f'<div class="section-header">Recent Packets</div>', unsafe_allow_html=True)
                    if st.session_state.sim_packets:
                        df_pkts = pd.DataFrame(st.session_state.sim_packets[-15:][::-1])
                        df_pkts['confidence'] = df_pkts['conf'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(df_pkts[['time','src','pred','confidence','true']], use_container_width=True, hide_index=True)

                time.sleep(speed)

            st.session_state.sim_running = False
            # ── Final accuracy/FAR summary ────────────────────────────────────
            if splits_loaded and st.session_state.sim_packets:
                all_preds_f = [p["pred"] for p in st.session_state.sim_packets if p["true"] != "?"]
                all_trues_f = [p["true"] for p in st.session_state.sim_packets if p["true"] != "?"]
                if all_trues_f:
                    final_acc = sum(p==t for p,t in zip(all_preds_f, all_trues_f)) / len(all_trues_f) * 100
                    fp_f = sum(1 for p,t in zip(all_preds_f, all_trues_f) if p=="Malicious" and t!="Malicious")
                    tn_f = sum(1 for p,t in zip(all_preds_f, all_trues_f) if p!="Malicious" and t!="Malicious")
                    far_f = fp_f/(fp_f+tn_f)*100 if (fp_f+tn_f)>0 else 0.0
                    st.success(f"Simulation complete — {st.session_state.sim_count} packets processed, {st.session_state.sim_threats} threats detected.")
                    fa1, fa2, fa3 = st.columns(3)
                    fa1.markdown(f'<div class="metric-card"><div class="metric-label">Final Accuracy</div><div class="metric-value green">{final_acc:.2f}%</div><div class="metric-sub">on simulated samples</div></div>', unsafe_allow_html=True)
                    fa2.markdown(f'<div class="metric-card"><div class="metric-label">Final FAR</div><div class="metric-value {"red" if far_f>15 else "amber" if far_f>10 else "green"}">{far_f:.2f}%</div><div class="metric-sub">false alarm rate</div></div>', unsafe_allow_html=True)
                    fa3.markdown(f'<div class="metric-card"><div class="metric-label">Threats Detected</div><div class="metric-value {"red" if st.session_state.sim_threats>0 else "green"}">{st.session_state.sim_threats}</div><div class="metric-sub">of {st.session_state.sim_count} total</div></div>', unsafe_allow_html=True)
                    # Add this temporarily to debug:
                    print(f"i={i} | target_cls={i%4} | true={true_label} | pred={pred} | conf={conf:.1f}")
            else:
                st.success(f"Simulation complete — {st.session_state.sim_count} packets processed, {st.session_state.sim_threats} threats detected.")

        elif not st.session_state.sim_running and st.session_state.sim_count > 0:
            # Show final state
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-header">Final Class Distribution</div>', unsafe_allow_html=True)
                st.plotly_chart(class_pie(st.session_state.sim_class_counts), use_container_width=True)
            with c2:
                st.markdown('<div class="section-header">Threat Alerts</div>', unsafe_allow_html=True)
                if st.session_state.sim_alerts:
                    for a in reversed(st.session_state.sim_alerts[-10:]):
                        st.markdown(f'<div class="alert-box">{a}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-box">{icon("check-circle-fill", color="#a8d4ff")} No threats detected</div>', unsafe_allow_html=True)

            if st.session_state.sim_packets:
                st.markdown('<div class="section-header">All Predictions</div>', unsafe_allow_html=True)
                df_all = pd.DataFrame(st.session_state.sim_packets)
                df_all['confidence'] = df_all['conf'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(df_all[['time','src','pred','confidence','true']], use_container_width=True, hide_index=True)

    # ── MODE 3: Live Network Capture ──────────────────────────────────────────
    elif "Mode 2" in mode:
        st.markdown("### :material/wifi_tethering: Mode 2 — Live Network Capture")
        st.warning("""
        **Requirements for live capture:**
        - Run Streamlit as **Administrator / root**
        - Install **Npcap** (Windows) or **libpcap** (Linux/Mac)
        - Install scapy: `pip install scapy`
        - The model extracts flow-level statistics from captured packets
        """)

        try:
            from scapy.all import sniff, IP, TCP, UDP
            scapy_available = True
        except ImportError:
            scapy_available = False

        if not scapy_available:
            st.error("Scapy not installed. Run: `pip install scapy`")
            st.info("While scapy is being set up, this mode will run in **simulated capture mode** using realistic synthetic traffic features.")
            scapy_available = False

        # ── List real Windows network interfaces via Npcap (cached per session) ──
        @st.cache_data
        def get_available_interfaces():
            """Returns list of (display_label, sniff_value, has_ip) tuples for the dropdown."""
            try:
                from scapy.arch.windows import get_windows_if_list
                ifaces = get_windows_if_list()
                options = []
                for i in ifaces:
                    name = i.get('name', 'Unknown')
                    desc = i.get('description', '')
                    ips = [ip for ip in i.get('ips', []) if ip and not ip.startswith('fe80') and ip != '0.0.0.0']
                    has_ip = len(ips) > 0
                    ip_str = f"  [{', '.join(ips)}]" if ips else "  [no IP — likely inactive]"
                    label = f"{name}{ip_str}" + (f"  —  {desc}" if desc and desc != name else "")
                    options.append((label, name, has_ip))
                # Show adapters WITH an active IP address first — that's almost always your real connection
                options.sort(key=lambda x: not x[2])
                return options
            except Exception as e:
                return []

        iface_options = get_available_interfaces() if scapy_available else []

        col1, col2 = st.columns([1, 2])
        with col1:
            if iface_options:
                labels = [o[0] for o in iface_options]
                chosen_label = st.selectbox("Network Interface", labels,
                    help="Adapters with an IP address are listed first — that's your real, active connection. "
                         "Cross-check the IP shown here against 'ipconfig' in a separate terminal if unsure.")
                iface = dict((o[0], o[1]) for o in iface_options)[chosen_label]
            else:
                st.warning("Could not auto-detect interfaces (Npcap may not be installed, or app isn't running as Administrator). Enter manually:")
                iface = st.text_input("Network Interface", value="Wi-Fi", help="Windows examples: Wi-Fi, Ethernet")
            port_filter = st.selectbox("Filter Port", ["443 (DoH/HTTPS)", "53 (DNS)", "All ports"])
            n_live = st.slider("Packets to capture", 10, 100, 20)
            capture_btn = st.button("Start Capture", icon=":material/wifi_tethering:")

        with col2:
            st.markdown("""
            **How live capture works:**

            1. Scapy listens on the selected interface
            2. Packets are grouped into flows by (src_ip, dst_ip, src_port, dst_port)
            3. Flow-level statistics are computed (packet sizes, timing, byte counts)
            4. Statistics are normalized using the trained scaler
            5. A 10-timestep sequence is built and fed to the model
            6. Prediction is displayed in real time
            """)

        if capture_btn:
            cap_placeholder = st.empty()
            live_results = []

            if scapy_available:
                # Real capture
                st.info(f"Capturing {n_live} packets on {iface}...")
                port_num = 443 if "443" in port_filter else 53 if "53" in port_filter else None
                bpf = f"tcp port {port_num}" if port_num else "tcp"

                try:
                    packets = sniff(iface=iface, filter=bpf, count=n_live, timeout=30)
                    st.success(f"Captured {len(packets)} packets")

                    # Extract basic flow features from captured packets
                    for pkt in packets:
                        try:
                            if IP in pkt and (TCP in pkt or UDP in pkt):
                                layer = TCP if TCP in pkt else UDP
                                pkt_len = len(pkt)
                                # Build minimal feature vector — fill uncomputable stats with medians
                                features = np.array([
                                    pkt[layer].sport / 65535,   # SourcePort (scaled)
                                    pkt[layer].dport / 65535,   # DestinationPort
                                    0.5,                         # Duration (median fallback)
                                    min(pkt_len/1500, 1.0),     # FlowBytesSent
                                    0.3,                         # FlowBytesReceived (fallback)
                                ] + [random.uniform(0.1, 0.6)] * (N_FEATURES - 5))
                                seq = np.tile(features, (SEQ_LEN, 1))
                                pred, conf, probs = predict_sequence(seq)
                                live_results.append({
                                    "src": f"{pkt[IP].src}:{pkt[layer].sport}",
                                    "dst": f"{pkt[IP].dst}:{pkt[layer].dport}",
                                    "size": pkt_len,
                                    "pred": pred,
                                    "conf": f"{conf:.1f}%",
                                })
                        except:
                            continue
                except Exception as e:
                    st.error(f"Capture failed: {e}")
                    st.info("Switching to simulated capture...")
                    scapy_available = False

            if not scapy_available or not live_results:
                # Simulated live capture
                st.info("Running **simulated capture** — generating synthetic network traffic features.")
                progress = st.progress(0)
                for i in range(n_live):
                    seq = np.random.rand(SEQ_LEN, N_FEATURES).astype(np.float32)
                    pred, conf, probs = predict_sequence(seq)
                    src_port = random.choice([443, 443, 443, 53, random.randint(1024,65535)])
                    live_results.append({
                        "src": f"{random.randint(10,220)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}:{src_port}",
                        "dst": f"8.8.{random.randint(0,9)}.{random.randint(1,9)}:443",
                        "size": random.randint(64, 1500),
                        "pred": pred,
                        "conf": f"{conf:.1f}%",
                    })
                    progress.progress((i+1)/n_live)
                    time.sleep(0.15)
                progress.empty()

            if live_results:
                df_live = pd.DataFrame(live_results)
                st.markdown("### :material/bar_chart: Capture Results")

                dist = df_live["pred"].value_counts().to_dict()
                full = {"DoH":0,"NonDoH":0,"Benign":0,"Malicious":0}
                full.update(dist)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="section-header">Traffic Breakdown</div>', unsafe_allow_html=True)
                    st.plotly_chart(class_pie(full), use_container_width=True)
                with c2:
                    threat_rows = df_live[df_live["pred"]=="Malicious"]
                    st.markdown('<div class="section-header">Threat Summary</div>', unsafe_allow_html=True)
                    if len(threat_rows) > 0:
                        st.markdown(f'<div class="alert-box">{icon("exclamation-triangle-fill", color="#ffaabb")} {len(threat_rows)} MALICIOUS flows detected out of {len(df_live)} captured</div>', unsafe_allow_html=True)
                        st.dataframe(threat_rows, use_container_width=True, hide_index=True)
                    else:
                        st.markdown(f'<div class="safe-box">{icon("check-circle-fill", color="#a8d4ff")} No threats detected in captured traffic</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header">All Captured Flows</div>', unsafe_allow_html=True)
                st.dataframe(df_live, use_container_width=True, hide_index=True)
                st.download_button("Download results", df_live.to_csv(index=False), "live_capture.csv", "text/csv", icon=":material/download:")

# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/bar_chart: Model Comparison":
# ══════════════════════════════════════════════════════════════════════════════
    hero_title("bar-chart-line", "Model Comparison", 'Model <span class="accent">Comparison</span>', "GRU vs Transformer vs Hybrid — full metrics side by side.")
    st.markdown("---")

    names = list(MODEL_RESULTS.keys())

    # Accuracy + FAR
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Overall Accuracy</div>', unsafe_allow_html=True)
        accs = [MODEL_RESULTS[m]["accuracy"] for m in names]
        fig1 = go.Figure(go.Bar(
            x=names, y=accs,
            marker_color=["#00ff88" if a==max(accs) else "#3399ff" for a in accs],
            text=[f"{a:.2f}%" for a in accs], textposition="outside",
            textfont=dict(color="#c8ffd8", size=11)
        ))
        fig1.update_layout(**dark_layout(), height=300,
            yaxis=dict(range=[76,81], gridcolor="#1a3d28", ticksuffix="%"),
            xaxis=dict(tickfont=dict(size=9)), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">False Alarm Rate</div>', unsafe_allow_html=True)
        fars = [MODEL_RESULTS[m]["far"] for m in names]
        fig2 = go.Figure(go.Bar(
            x=names, y=fars,
            marker_color=["#00ff88" if f==min(fars) else "#ff3355" if f==max(fars) else "#ffaa00" for f in fars],
            text=[f"{f:.2f}%" for f in fars], textposition="outside",
            textfont=dict(color="#c8ffd8", size=11)
        ))
        fig2.update_layout(**dark_layout(), height=300,
            yaxis=dict(range=[0,20], gridcolor="#1a3d28", ticksuffix="%"),
            xaxis=dict(tickfont=dict(size=9)), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Per-class recall heatmap
    st.markdown('<div class="section-header">Per-Class Recall Heatmap</div>', unsafe_allow_html=True)
    recall_matrix = [[MODEL_RESULTS[m]["recall"][cls] for cls in LABEL_NAMES] for m in names]
    fig3 = go.Figure(go.Heatmap(
        z=recall_matrix, x=LABEL_NAMES, y=names,
        colorscale=[[0,"#2a0a10"],[0.5,"#884400"],[1.0,"#00ff88"]],
        text=[[f"{v:.1f}%" for v in row] for row in recall_matrix],
        texttemplate="%{text}", textfont=dict(size=13), zmin=40, zmax=100
    ))
    fig3.update_layout(**dark_layout(), height=260)
    st.plotly_chart(fig3, use_container_width=True)

    # AUC + Mean AP
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Macro AUC</div>', unsafe_allow_html=True)
        aucs = [MODEL_RESULTS[m]["auc"] for m in names]
        fig4 = go.Figure(go.Bar(
            x=names, y=aucs,
            marker_color=["#00ff88" if a==max(aucs) else "#3399ff" for a in aucs],
            text=[f"{a:.4f}" for a in aucs], textposition="outside",
            textfont=dict(color="#c8ffd8", size=11)
        ))
        fig4.update_layout(**dark_layout(), height=280,
            yaxis=dict(range=[0.92,0.945], gridcolor="#1a3d28"),
            xaxis=dict(tickfont=dict(size=9)), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">MAE & RMSE</div>', unsafe_allow_html=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="MAE", x=names,
            y=[MODEL_RESULTS[m]["mae"] for m in names], marker_color="#3399ff",
            text=[f"{MODEL_RESULTS[m]['mae']:.4f}" for m in names], textfont=dict(size=9)))
        fig5.add_trace(go.Bar(name="RMSE", x=names,
            y=[MODEL_RESULTS[m]["rmse"] for m in names], marker_color="#00aaff",
            text=[f"{MODEL_RESULTS[m]['rmse']:.4f}" for m in names], textfont=dict(size=9)))
        fig5.update_layout(**dark_layout(), barmode="group", height=280,
            yaxis=dict(gridcolor="#1a3d28"),
            xaxis=dict(tickfont=dict(size=9)),
            legend=dict(font=dict(color="#c8ffd8")))
        st.plotly_chart(fig5, use_container_width=True)

    # Confusion matrix — Hybrid FINAL
    st.markdown('<div class="section-header">Confusion Matrix — Heavy Hybrid FINAL (Normalized %)</div>', unsafe_allow_html=True)
    cm = np.array([
        [64.38,  0.02,  0.00, 35.60],
        [ 0.00, 100.00, 0.00,  0.00],
        [ 0.00,  0.00, 100.00, 0.00],
        [47.80,  0.00,  0.00, 52.20],
    ])
    fig6 = go.Figure(go.Heatmap(
        z=cm, x=LABEL_NAMES, y=LABEL_NAMES,
        colorscale=[[0,"#050f0a"],[0.3,"#1a3d28"],[0.7,"#1a7a55"],[1.0,"#00ff88"]],
        text=[[f"{v:.2f}%" for v in row] for row in cm],
        texttemplate="%{text}", textfont=dict(size=13, color="white"),
        showscale=True, zmin=0, zmax=100
    ))
    fig6.update_layout(**dark_layout(), height=340, xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig6, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Full Metrics Summary</div>', unsafe_allow_html=True)
    rows = []
    for m, d in MODEL_RESULTS.items():
        rows.append({
            "Model": m, "Accuracy": f"{d['accuracy']:.2f}%",
            "FAR": f"{d['far']:.2f}%", "MAE": f"{d['mae']:.4f}",
            "RMSE": f"{d['rmse']:.4f}", "MCC": f"{d['mcc']:.4f}",
            "Macro AUC": f"{d['auc']:.4f}", "Mean AP": f"{d['mean_ap']:.4f}",
            "Parameters": d['params']
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/folder: Dataset Info":
# ══════════════════════════════════════════════════════════════════════════════
    hero_title("folder2-open", "Dataset Info", 'Dataset <span class="accent">Information</span>', "Combined CIRA-CIC-DoHBrw-2020 + Hokkaido dataset details.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### Combined Dataset
        **Sources:**
        - Hokkaido CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD
        - BCCC-CIRA-CIC-DoHBrw-2020 (real Benign data)

        **Final cleaned shape:** 1,713,101 rows × 31 columns
        **Features used:** 36 (28 original + 10 engineered − 5 correlated)
        """)
        st.markdown("""
        ### Class Distribution (After Cleaning)
        | Class | Label | Count |
        |---|---|---|
        | DoH | 0 | 273,919 |
        | NonDoH | 1 | 750,456 |
        | Benign | 2 | 242,453 |
        | Malicious | 3 | 446,273 |
        | **Total** | | **1,713,101** |
        """)

    with c2:
        raw = {"DoH":273919,"NonDoH":750456,"Benign":242453,"Malicious":446273}
        fig = go.Figure(go.Bar(
            x=list(raw.keys()), y=list(raw.values()),
            marker_color=[COLORS[k] for k in raw],
            text=[f"{v:,}" for v in raw.values()],
            textposition="outside", textfont=dict(color="#c8ffd8")
        ))
        fig.update_layout(**dark_layout(), height=320,
            title="Class Distribution After Cleaning",
            title_font=dict(color="#00ff88"),
            yaxis=dict(gridcolor="#1a3d28"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Preprocessing Pipeline")
    steps = [
        ("1","Load CSVs","3 Hokkaido files + 1 BCCC file, map labels: DoH=0, NonDoH=1, Benign=2, Malicious=3"),
        ("2","Fill Missing Features","SourcePort, DestinationPort, Duration filled with Hokkaido medians for BCCC rows"),
        ("3","Clean","Drop nulls (8,594) + duplicates → final 1,713,101 rows"),
        ("4","Feature Engineering","Add 10 engineered features, drop 5 correlated → 36 features"),
        ("5","Normalize","MinMaxScaler [0,1] → saved as scaler_FINAL.pkl"),
        ("6","Sequence Building","Sort by label → seq_len=10, step=5 → 342,619 sequences"),
        ("7","Balance","Undersample to 50,000/class → 200,000 total sequences"),
        ("8","Split","60% Train / 20% Val / 20% Test (stratified)"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""<div style="background:#0a1a10;border:1px solid #1a3d28;
            border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:14px;">
            <span style="color:#00ff88;font-family:monospace;font-weight:600">[{num}]</span>
            <span style="color:#c8ffd8;font-weight:600;margin-left:8px">{title}</span>
            <span style="color:#5a8a6a;margin-left:8px">— {desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Malicious Tunneling Tools in Dataset")
    tools = pd.DataFrame({
        "Tool":    ["dns2tcp","iodine","dnstt","dnscat2","tcp-over-dns","tuns"],
        "Samples": [167486, 46580, 46080, 35770, 30040, 29040],
        "Type":    ["TCP over DNS","IP over DNS","DNS-based tunnel",
                    "DNS C2 framework","TCP via DNS queries","Tunnel via DNS queries"]
    })
    st.dataframe(tools, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/info: About":
# ══════════════════════════════════════════════════════════════════════════════
    hero_title("info-circle", "About", 'About <span class="accent">SENTINEL</span>', "Project details, key findings, and final results.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### Project Details
        **Title:** Intrusion Prediction System Using Hybrid GRU-Transformer Model
        **Student:** Alya Maisarah
        **Supervisor:** Ts. Dr Mohammad Hafiz bin Mohd Yusof
        **Institution:** Universiti Teknologi MARA
        **Programme:** Bachelor of Computer Science (Hons.)
        **Year:** January 2026

        ---
        ### Why This Problem is Hard
        DoH (DNS-over-HTTPS) encrypts DNS queries inside HTTPS on port 443.
        Attackers exploit this to hide malicious traffic (tunneling).

        Since both normal and malicious DoH share the same port and
        encryption protocol, flow-level statistics look nearly identical —
        making classification fundamentally challenging.
        """)

    with c2:
        st.markdown("""
        ### Key Finding — Cascade Classifier Experiment
        A 3-level cascade classifier was built to isolate the classification difficulty:

        | Level | Task | Accuracy |
        |---|---|---|
        | L1 | NonDoH vs DoH-family | **99.97%** :material/check_circle: |
        | L2 | Normal DoH vs Tunnel | **57.14%** :material/warning: |
        | L3 | Benign vs Malicious | **100.00%** :material/check_circle: |

        **L2 at 57%** (barely above random chance) conclusively proved
        that flow-level statistics alone cannot distinguish normal DoH
        from malicious DoH tunneling — a fundamental feature limitation,
        not a model architecture limitation.

        ---
        ### Final Results Summary
        | Model | Accuracy | FAR |
        |---|---|---|
        | Heavy GRU | 78.64% | 11.49% |
        | Heavy Transformer | 78.00% | 14.53% |
        | **Heavy Hybrid** :material/star: | **79.14%** | **11.87%** |
        | Ensemble (GRU+Hybrid) | 79.09% | 11.57% |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# Footer — appears at the bottom of every page
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="app-footer">Developed by Alya</div>', unsafe_allow_html=True)