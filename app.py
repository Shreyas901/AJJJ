"""
Avenue Dataset — Abnormal Event Detection
Streamlit Application
"""

import os
import cv2
import numpy as np
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Abnormal Event Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0a0a0a; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #f5f5f5; letter-spacing: -1px; }
.hero-sub { font-size: 0.9rem; color: #555; font-family: 'Space Mono', monospace; }
.metric-card { background: #141414; border: 1px solid #242424; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #f5f5f5; }
.metric-label { font-size: 0.72rem; color: #555; text-transform: uppercase; }
.metric-value.red { color: #ff4757; }
.metric-value.green { color: #2ed573; }
.metric-value.amber { color: #ffa502; }
.verdict-wrap { text-align: center; padding: 1.5rem 0; }
.verdict-badge { display: inline-block; padding: 0.6rem 2rem; border-radius: 50px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.2rem; }
.v-abn { background: #ff475715; color: #ff4757; border: 2px solid #ff4757; }
.v-nrm { background: #2ed57315; color: #2ed573; border: 2px solid #2ed573; }
.sec-hdr { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #444; text-transform: uppercase; letter-spacing: 3px; border-bottom: 1px solid #1e1e1e; padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem 0; }
.info-box { background: #0f0f0f; border-left: 3px solid #ffa502; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; font-size: 0.84rem; color: #888; margin: 1rem 0; }
.path-box { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 8px; padding: 0.6rem 1rem; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #2ed573; margin: 0.5rem 0; }
[data-testid="stSidebar"] { background: #0d0d0d; border-right: 1px solid #1a1a1a; }
.stButton > button { background: #f5f5f5; color: #0a0a0a; font-family: 'Space Mono', monospace; font-weight: 700; border: none; border-radius: 8px; padding: 0.65rem 1.5rem; width: 100%; }
.stButton > button:hover { background: #ffa502; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "mobilenet_lstm_weights.weights.h5")
SEQ_LENGTH = 16

@st.cache_resource(show_spinner="🔄 Loading models...")
def load_models(weights_path):
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = Input(shape=(224, 224, 3))
    x = preprocess_input(inp)
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    mobilenet = Model(inp, x, name='MobileNetV2_Extractor')
    
    inputs = Input(shape=(SEQ_LENGTH, 1280))
    x = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(inputs)
    x = Dropout(0.4)(x)
    x = LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    lstm_model = Model(inputs, outputs)
    
    lstm_model.load_weights(weights_path)
    
    return lstm_model, mobilenet

def extract_frames(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frames, idx = [], 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % frame_skip == 0:
            resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
            frames.append(resized.astype(np.float32))
        idx += 1
    cap.release()
    return frames, total, fps, w, h

def get_mobilenet_features(frames, extractor, batch_size=32):
    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = np.array(frames[i:i+batch_size])
        feats = extractor.predict(batch, verbose=0)
        all_feats.append(feats)
    return np.vstack(all_feats)

def lstm_predict(features, lstm_model, seq_length=16, step=8, threshold=0.5):
    if len(features) < seq_length:
        return np.array([]), np.array([])
    seqs, starts = [], []
    for s in range(0, len(features) - seq_length + 1, step):
        seqs.append(features[s:s+seq_length])
        starts.append(s)
    seq_probs = lstm_model.predict(np.array(seqs), verbose=0).flatten()
    accum = np.zeros(len(features))
    count = np.zeros(len(features))
    for i, s in enumerate(starts):
        accum[s:s+seq_length] += seq_probs[i]
        count[s:s+seq_length] += 1
    probs = accum / np.maximum(count, 1)
    labels = (probs > threshold).astype(int)
    return probs, labels

def full_pipeline(video_path, lstm_model, mob_extractor, frame_skip, seq_length, step, threshold, pbar):
    pbar.progress(5, "📂 Reading frames...")
    frames, total, fps, w, h = extract_frames(video_path, frame_skip)
    if len(frames) == 0: return None
    pbar.progress(30, "🧠 Extracting features...")
    features = get_mobilenet_features(frames, mob_extractor)
    pbar.progress(65, "⚡ LSTM inference...")
    probs, labels = lstm_predict(features, lstm_model, seq_length, step, threshold)
    pbar.progress(95, "📊 Computing stats...")
    pct = 100.0 * labels.mean() if len(labels) > 0 else 0.0
    pbar.progress(100, "✅ Done!")
    return {
        "frames": frames, "total": total, "fps": fps, "width": w, "height": h,
        "features": features, "probs": probs, "labels": labels, "pct": pct,
        "mean_p": float(probs.mean()) if len(probs) > 0 else 0,
        "max_p": float(probs.max()) if len(probs) > 0 else 0,
        "min_p": float(probs.min()) if len(probs) > 0 else 0,
        "n_abn": int(labels.sum()), "n_nrm": int((labels == 0).sum()),
        "verdict": "ABNORMAL" if pct > 10 else "NORMAL",
        "duration": total / fps if fps > 0 else 0,
    }

def plot_timeline(probs, labels, threshold, title):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor('#0a0a0a'); ax.set_facecolor('#111')
    t = np.arange(len(probs))
    in_abn, s_abn = False, 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_abn: s_abn = i; in_abn = True
        elif labels[i] == 0 and in_abn: ax.axvspan(s_abn, i, alpha=0.18, color='#ff4757'); in_abn = False
    if in_abn: ax.axvspan(s_abn, len(labels), alpha=0.18, color='#ff4757')
    ax.fill_between(t, 0, probs, where=(probs > threshold), alpha=0.4, color='#ff4757')
    ax.fill_between(t, 0, probs, where=(probs <= threshold), alpha=0.15, color='#2ed573')
    ax.plot(t, probs, color='#ffa502', lw=1.8, label='P(Abnormal)')
    ax.axhline(threshold, color='#fff', lw=1.2, ls='--', alpha=0.4, label=f'Threshold = {threshold}')
    ax.set_xlim(0, len(probs)); ax.set_ylim(0, 1)
    ax.set_xlabel('Frame Index', color='#555', fontsize=9); ax.set_ylabel('Anomaly Probability', color='#555', fontsize=9)
    ax.set_title(title, color='#f0f0f0', fontsize=10, pad=8, fontfamily='monospace')
    ax.tick_params(colors='#333', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#222')
    ax.legend(fontsize=8, facecolor='#141414', edgecolor='#222', labelcolor='#aaa', loc='upper right')
    plt.tight_layout()
    return fig

def plot_frame_grid(frames, probs, labels, threshold, n=10):
    n_avail = min(len(frames), len(probs))
    if n_avail == 0: return None
    indices = np.linspace(0, n_avail-1, min(n, n_avail), dtype=int)
    cols = 5; rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.8, rows*3.0))
    fig.patch.set_facecolor('#0a0a0a')
    if rows == 1: axes = np.array([axes])
    axes_flat = axes.flatten()
    for pi, fi in enumerate(indices):
        ax = axes_flat[pi]
        ax.imshow(frames[fi].astype(np.uint8))
        p = probs[fi]; v = 'ABN' if p > threshold else 'NRM'
        c = '#ff4757' if v == 'ABN' else '#2ed573'
        ax.set_title(f'Frame {fi}\np={p:.3f}\n{v}', fontsize=7, color=c, fontweight='bold', pad=3)
        for sp in ax.spines.values(): sp.set_edgecolor(c); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('#111')
    for j in range(pi+1, len(axes_flat)): axes_flat[j].set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    model_exists = os.path.exists(WEIGHTS_PATH)
    status_color = "#2ed573" if model_exists else "#ff4757"
    st.markdown(f"<div style='font-size:0.7rem;color:{status_color}'>{'✅ Model found' if model_exists else '❌ Model not found'}</div>", unsafe_allow_html=True)
    threshold = st.slider("Anomaly Threshold", 0.1, 0.9, 0.5, 0.05)
    frame_skip = st.slider("Frame Skip", 1, 10, 5, 1)
    seq_length = st.select_slider("Sequence Length", [8, 12, 16, 24, 32], value=16)
    step_size = st.slider("Sliding Window Step", 1, 16, 8, 1)
    n_show = st.slider("Sample Frames", 5, 20, 10, 1)

st.markdown("<div class='hero-title'>🎥 Abnormal Event Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>MobileNetV2 (features) → LSTM → Sigmoid</div>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>Upload any video (.mp4, .avi, .mov) to detect abnormal events. Frames with probability above threshold are classified as <b style='color:#ff4757'>ABNORMAL</b>.</div>", unsafe_allow_html=True)

if not os.path.exists(WEIGHTS_PATH):
    st.error(f"❌ Model weights not found at {WEIGHTS_PATH}")
    st.stop()

lstm_model, mob_extractor = load_models(WEIGHTS_PATH)
st.success("✅ Model loaded | MobileNetV2 extractor ready")

uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded.read()); tmp_file.flush()
    video_path = tmp_file.name
    
    run = st.button("▶ RUN ANOMALY DETECTION", use_container_width=True)
    
    if run:
        st.session_state["result"] = None
        st.session_state["video_name"] = uploaded.name
    
    if run or st.session_state.get("result") is not None:
        if run:
            pbar = st.progress(0, "Starting...")
            result = full_pipeline(video_path, lstm_model, mob_extractor, frame_skip, seq_length, step_size, threshold, pbar)
            st.session_state["result"] = result
        else:
            result = st.session_state["result"]
        
        if result is None:
            st.error("❌ Could not process video."); st.stop()
        
        vname = st.session_state.get("video_name", uploaded.name)
        st.markdown("---")
        v_cls = "v-abn" if result["verdict"] == "ABNORMAL" else "v-nrm"
        st.markdown(f"<div class='verdict-wrap'><div class='verdict-badge {v_cls}'>{'⚠️' if result['verdict']=='ABNORMAL' else '✅'} {result['verdict']}</div></div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metrics = [
            (c1, str(len(result["probs"])), "Frames Analyzed", ""),
            (c2, str(result["n_nrm"]), "Normal Frames", "green"),
            (c3, str(result["n_abn"]), "Abnormal Frames", "red"),
            (c4, f"{result['pct']:.1f}%", "% Abnormal", "amber"),
            (c5, f"{result['mean_p']:.3f}", "Mean Prob", "amber"),
            (c6, f"{result['max_p']:.3f}", "Max Prob", "red"),
        ]
        for col, val, lbl, cls in metrics:
            with col:
                st.markdown(f"<div class='metric-card'><div class='metric-value {cls}'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 TIMELINE", "🖼️ FRAMES", "📊 DISTRIBUTION", "📋 DATA"])
        
        with tab1:
            fig = plot_timeline(result["probs"], result["labels"], threshold, f"{vname} - Anomaly Probability")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        
        with tab2:
            fig = plot_frame_grid(result["frames"], result["probs"], result["labels"], threshold, n_show)
            if fig: st.pyplot(fig, use_container_width=True); plt.close(fig)
        
        with tab3:
            st.dataframe(pd.DataFrame({"Frame": np.arange(len(result["probs"])), "Probability": np.round(result["probs"], 4), "Label": ["ABNORMAL" if l else "NORMAL" for l in result["labels"]]}), use_container_width=True)
        
        with tab4:
            df = pd.DataFrame({"Frame Index": np.arange(len(result["probs"])), "Probability": np.round(result["probs"], 4), "Label": ["ABNORMAL" if l else "NORMAL" for l in result["labels"]]})
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv, file_name=f"{vname}_predictions.csv", mime="text/csv")
    
    try: os.unlink(video_path)
    except: pass
