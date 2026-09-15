import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Product Defect Quality Classifier",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Manufacturing QA Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #701A75 100%);
        border: 1px solid rgba(232, 121, 249, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(232, 121, 249, 0.15);
        color: #F0ABFC;
        border: 1px solid rgba(232, 121, 249, 0.35);
        margin-bottom: 10px;
    }

    .qa-card-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .qa-card-fail {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(185, 28, 28, 0.06) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .qa-hero {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 6px 0;
    }

    .action-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #E879F9;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Product_Defect_Classification_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("product_defect_classifier.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Smart Factory & Inline Quality Assurance AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">⚙️ Product Defect Quality Classifier</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Screen manufactured components in real-time as Defective or Non-Defective using machine telemetry, line speed, material quality, and operator expertise.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Inline QA Inspection", "📁 Batch Production Lot Screening (CSV)", "📊 Model Performance & Confusion Matrix"])

# --- TAB 1: Inline QA Inspection ---
with tabs[0]:
    st.subheader("Assembly Line Telemetry & Quality Sensor Feed")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_pass = st.button("🟢 QA Certified (Non-Defective)", width="stretch")
    with p_cols[1]:
        load_fail = st.button("🔴 Critical Out-of-Spec (Defective)", width="stretch")
    with p_cols[2]:
        load_sample = st.button("📋 Sample Component", width="stretch")

    if load_pass:
        st.session_state["temp"] = 68.0
        st.session_state["press"] = 4.2
        st.session_state["vib"] = 1.2
        st.session_state["speed"] = 45.0
        st.session_state["quality"] = 9.2
        st.session_state["exp"] = 14
    elif load_fail:
        st.session_state["temp"] = 112.0
        st.session_state["press"] = 8.5
        st.session_state["vib"] = 7.8
        st.session_state["speed"] = 92.0
        st.session_state["quality"] = 3.5
        st.session_state["exp"] = 2
    elif load_sample:
        st.session_state["temp"] = 82.0
        st.session_state["press"] = 5.6
        st.session_state["vib"] = 4.5
        st.session_state["speed"] = 60.0
        st.session_state["quality"] = 6.8
        st.session_state["exp"] = 8

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 🌡️ Equipment Physical Sensors")
        temperature = st.slider(
            "Curing / Mold Temperature (°C)", 40.0, 130.0,
            value=float(st.session_state.get("temp", 82.0)), step=0.5,
            key="input_temp"
        )
        pressure = st.slider(
            "Hydraulic / Clamping Pressure (bar)", 1.0, 10.0,
            value=float(st.session_state.get("press", 5.6)), step=0.1,
            key="input_press"
        )
        vibration = st.slider(
            "Spindle Vibration Level (0 - 10)", 0.0, 10.0,
            value=float(st.session_state.get("vib", 4.5)), step=0.1,
            key="input_vib", help="High vibration leads to micro-cracks and surface roughness."
        )

    with c_right:
        st.markdown("#### ⚡ Line Velocity & Raw Material")
        speed = st.slider(
            "Production Line Speed (Units / Min)", 10.0, 100.0,
            value=float(st.session_state.get("speed", 60.0)), step=1.0,
            key="input_speed"
        )
        quality = st.slider(
            "Raw Material Quality Score (1 - 10)", 1.0, 10.0,
            value=float(st.session_state.get("quality", 6.8)), step=0.1,
            key="input_quality", help="Certified raw polymer/alloy feedstock purity benchmark."
        )
        experience = st.slider(
            "Operator Work Experience (Years)", 0, 20,
            value=int(st.session_state.get("exp", 8)), step=1,
            key="input_exp"
        )

    st.markdown("---")
    eval_btn = st.button("🚀 Evaluate Component QA Status", type="primary", width="stretch")

    component_df = pd.DataFrame([{
        "temperature_c": temperature,
        "pressure_bar": pressure,
        "vibration_level": vibration,
        "production_speed": speed,
        "material_quality_score": quality,
        "operator_experience_years": experience
    }])

    pred = model.predict(component_df)[0]
    probs = model.predict_proba(component_df)[0]
    classes = list(model.classes_)
    conf = max(probs) * 100

    def_idx = classes.index("Defective") if "Defective" in classes else 0
    def_prob = probs[def_idx] * 100

    st.markdown("### 📋 Inline Quality Inspection Verdict")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        if pred == "Defective":
            st.markdown(f"""
            <div class="qa-card-fail">
                <span style="font-size: 2.8rem;">🚨</span>
                <div style="color: #F87171; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    REJECT: Defect Detected
                </div>
                <div class="qa-hero" style="color: #EF4444;">
                    {conf:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Defect Confidence Probability</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="qa-card-pass">
                <span style="font-size: 2.8rem;">✅</span>
                <div style="color: #34D399; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    PASS: Quality Certified
                </div>
                <div class="qa-hero" style="color: #10B981;">
                    {conf:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Certification Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        st.write(f"**Defect Likelihood:** {def_prob:.1f}%")
        st.progress(float(def_prob / 100.0))

    with r2:
        st.markdown("#### 🔍 Root Cause Diagnostics")
        alerts = []
        if vibration >= 5.5:
            alerts.append(("Harmonic Vibration Spike", f"Vibration ({vibration:.1f}) exceeds mechanical tolerance; causes surface abrasions.", "fail"))
        if quality <= 5.0:
            alerts.append(("Substandard Feedstock", f"Material purity score ({quality:.1f}) compromises structural tensile strength.", "fail"))
        if temperature >= 100.0:
            alerts.append(("Overheating Mold Hazard", f"Temperature at {temperature:.1f}°C causes thermal warping.", "fail"))
        if speed >= 80.0:
            alerts.append(("Line Velocity Overrun", f"High throughput ({speed:.0f} u/min) exceeds optimal cooling dwell window.", "warn"))

        if alerts:
            for title, desc, tone in alerts:
                if tone == "fail":
                    st.error(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.success("🎉 **Tolerances Met**: All sensor channels conform to Six Sigma manufacturing limits.")

        st.markdown(f"""
        <div class="action-box">
            <strong style="color: #F0ABFC;">QA Engineering Directive:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'DIVERT TO QUARANTINE: Reject unit for destructive testing. Recalibrate spindle tension and inspect feed batch.' if pred == 'Defective' else 'RELEASE TO PACKAGING: Component passed all automated optical and structural inspection criteria.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Telemetry Vector"):
        st.dataframe(component_df, width="stretch")

# --- TAB 2: Batch Production Screening ---
with tabs[1]:
    st.subheader("Batch Production Lot Quality Screening")
    st.write("Upload a manufacturing run CSV or screen against the 600-component baseline production dataset.")

    csv_file = st.file_uploader("Upload Production Lot CSV", type=["csv"], key="defect_csv")
    df_lot = None

    if csv_file is not None:
        df_lot = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_lot)} component records from file.")
    else:
        sample_path = get_asset_path("data/product_defects.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline production lot (`data/product_defects.csv`)", value=True):
                df_lot = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_lot)} records from baseline production run.")

    if df_lot is not None:
        req_cols = ["temperature_c", "pressure_bar", "vibration_level", "production_speed", "material_quality_score", "operator_experience_years"]
        missing = [c for c in req_cols if c not in df_lot.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Screen Entire Production Lot", type="primary"):
                with st.spinner("Classifying lot quality..."):
                    preds = model.predict(df_lot[req_cols])
                    probs = model.predict_proba(df_lot[req_cols])
                    d_idx = list(model.classes_).index("Defective") if "Defective" in list(model.classes_) else 0
                    defect_probs = probs[:, d_idx] * 100

                    res_df = df_lot.copy()
                    res_df["QA_Verdict"] = preds
                    res_df["Defect_Probability_%"] = np.round(defect_probs, 1)

                    def_count = sum(preds == "Defective")
                    pass_count = sum(preds == "Non-Defective")
                    def_rate = (def_count / len(res_df)) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Components", len(res_df))
                    m2.metric("Defects Rejected", def_count, delta=f"{def_rate:.1f}% Scrap Rate", delta_color="inverse")
                    m3.metric("QA Certified (Passed)", pass_count)
                    m4.metric("Average Defect Prob", f"{np.mean(defect_probs):.1f}%")

                    f_choice = st.radio("Filter Production Table:", ["All Units", "Defective Only", "Non-Defective Only"], horizontal=True)
                    if f_choice == "Defective Only":
                        view = res_df[res_df["QA_Verdict"] == "Defective"]
                    elif f_choice == "Non-Defective Only":
                        view = res_df[res_df["QA_Verdict"] == "Non-Defective"]
                    else:
                        view = res_df

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download QA Inspection Lot Report as CSV",
                        data=csv_export,
                        file_name="product_defect_screening_report.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Classification Benchmark")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 QA Classifier Specifications
        - **Algorithm**: `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`
        - **Monitored Telemetry**:
            - Thermal & Pneumatic: `temperature_c`, `pressure_bar`
            - Mechanical Dynamics: `vibration_level`
            - Operations: `production_speed`, `material_quality_score`, `operator_experience_years`
        - **Accuracy Benchmark**:
            - **Overall Accuracy**: **90.00%** on Stratified Test Split
            - High recall on defective parts (~0.93) to minimize customer returns.
        """)

    with c2:
        img_path = get_asset_path("confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Confusion Matrix on Test Split", width="stretch")
        else:
            st.info("Confusion matrix image not found.")

st.caption("Smart Factory & Inline Defect Classification • Scikit-learn & Streamlit")
