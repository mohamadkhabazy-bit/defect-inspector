# %%
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.detector import DefectDetector, CLASS_NAMES, CLASS_COLORS
from utils.heatmap import DefectHeatmap

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Defect Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }
    .stApp {
        background-color: #0d0f14;
        color: #e0e6f0;
    }
    h1, h2, h3 {
        font-family: 'Share Tech Mono', monospace;
        color: #00e5ff;
        letter-spacing: 2px;
    }
    .stSidebar {
        background-color: #0a0c10 !important;
    }
    .stButton>button {
        background: #00e5ff22;
        border: 1px solid #00e5ff;
        color: #00e5ff;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
        border-radius: 2px;
    }
    .stButton>button:hover {
        background: #00e5ff44;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Share Tech Mono', monospace;
        color: #00e5ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "detector" not in st.session_state:
    st.session_state.detector = None
if "heatmap" not in st.session_state:
    st.session_state.heatmap = DefectHeatmap(width=200, height=200)
if "total_defects" not in st.session_state:
    st.session_state.total_defects = 0
if "images_scanned" not in st.session_state:
    st.session_state.images_scanned = 0
if "class_totals" not in st.session_state:
    st.session_state.class_totals = {cls: 0 for cls in CLASS_NAMES}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CONFIG")
    st.markdown("---")

    model_path = st.text_input(
        "Model weights path",
        value="model/best.pt",
        help="Path to your trained YOLOv11 .pt file",
    )
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)

    load_btn = st.button("⚡ Load Model", use_container_width=True)
    if load_btn:
        with st.spinner("Loading YOLOv11..."):
            try:
                st.session_state.detector = DefectDetector(
                    model_path=model_path,
                    conf=conf_threshold,
                )
                st.success("Model loaded ✓")
            except Exception as e:
                st.error(f"Failed to load model:\n{e}")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.metric("Images Scanned", st.session_state.images_scanned)
    st.metric("Total Defects",  st.session_state.total_defects)

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.heatmap        = DefectHeatmap()
        st.session_state.total_defects  = 0
        st.session_state.images_scanned = 0
        st.session_state.class_totals   = {cls: 0 for cls in CLASS_NAMES}
        if st.session_state.detector:
            st.session_state.detector.reset()
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 DEFECT INSPECTOR")
st.markdown("*YOLOv11 · NEU Surface Defect Dataset · Industrial Vision System*")
st.markdown("---")

# ── Model status ───────────────────────────────────────────────────────────────
if st.session_state.detector is None:
    st.warning("⚠️ No model loaded. Use the sidebar to load your YOLOv11 weights.")
else:
    st.success("✅ Model ready — upload an image to begin inspection.")

# ── Image upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a NEU dataset image or any surface image for inspection.",
)

if uploaded and st.session_state.detector:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_h, img_w = img_bgr.shape[:2]

    with st.spinner("🔍 Scanning for defects..."):
        annotated, detections = st.session_state.detector.predict(img_bgr)

    n_found = len(detections)
    st.session_state.total_defects  += n_found
    st.session_state.images_scanned += 1

    for det in detections:
        st.session_state.class_totals[det["class"]] += 1

    centers = [det["center"] for det in detections]
    st.session_state.heatmap.update(centers, img_w, img_h)

    # ── Main display ───────────────────────────────────────────────────────────
    col_img, col_heat = st.columns([1, 1])

    with col_img:
        st.markdown("### 🖼 Detection Result")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    with col_heat:
        st.markdown("### 🌡 Spatial Heatmap")
        heatmap_rgb = st.session_state.heatmap.render_rgb(output_size=(400, 400))
        st.image(heatmap_rgb, use_container_width=True)
        st.caption("Accumulates defect locations across all inspected images.")

    st.markdown("---")

    # ── Metrics ────────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Defects this image",    n_found)
    m2.metric("Total session defects", st.session_state.total_defects)
    m3.metric("Images scanned",        st.session_state.images_scanned)
    top_cls = max(st.session_state.class_totals, key=st.session_state.class_totals.get)
    m4.metric("Most common defect",    top_cls)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns([1, 1])

    with ch1:
        st.markdown("### 📊 Defect Class Frequency")
        df_counts = pd.DataFrame({
            "Class": list(st.session_state.class_totals.keys()),
            "Count": list(st.session_state.class_totals.values()),
        }).sort_values("Count", ascending=True)

        bar_colors = []
        for cls in df_counts["Class"]:
            b, g, r = CLASS_COLORS.get(cls, (150, 150, 150))
            bar_colors.append(f"rgb({r},{g},{b})")

        fig_bar = go.Figure(go.Bar(
            x=df_counts["Count"],
            y=df_counts["Class"],
            orientation="h",
            marker_color=bar_colors,
            text=df_counts["Count"],
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#141820",
            font_color="#e0e6f0",
            font_family="Share Tech Mono",
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            margin=dict(l=10, r=30, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        st.markdown("### 🥧 Class Distribution")
        pie_labels = [k for k, v in st.session_state.class_totals.items() if v > 0]
        pie_values = [v for v in st.session_state.class_totals.values() if v > 0]

        if pie_values:
            pie_colors = []
            for cls in pie_labels:
                b, g, r = CLASS_COLORS.get(cls, (150, 150, 150))
                pie_colors.append(f"rgb({r},{g},{b})")

            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker_colors=pie_colors,
                hole=0.4,
                textfont_family="Share Tech Mono",
            ))
            fig_pie.update_layout(
                paper_bgcolor="#0d0f14",
                font_color="#e0e6f0",
                font_family="Share Tech Mono",
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                legend=dict(bgcolor="#141820"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No defects detected yet.")

    # ── Confidence strip ───────────────────────────────────────────────────────
    if detections:
        st.markdown("### 📈 Confidence Scores — This Image")
        conf_data = pd.DataFrame(detections)[["class", "confidence"]]
        fig_conf = px.strip(
            conf_data, x="confidence", y="class", color="class",
            color_discrete_map={
                cls: f"rgb({r},{g},{b})"
                for cls, (b, g, r) in CLASS_COLORS.items()
            },
        )
        fig_conf.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#141820",
            font_color="#e0e6f0",
            font_family="Share Tech Mono",
            xaxis=dict(range=[0, 1], gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            showlegend=False,
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Detection log ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Detection Log")

    df_log = pd.DataFrame(st.session_state.detector.log)

    if not df_log.empty:
        df_log = df_log[["time", "class", "confidence", "center"]]
        st.dataframe(
            df_log.sort_index(ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )

        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Log as CSV",
            data=csv,
            file_name="defect_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No detections logged yet.")

elif uploaded and st.session_state.detector is None:
    st.error("Please load a model first using the sidebar.")

# %%



