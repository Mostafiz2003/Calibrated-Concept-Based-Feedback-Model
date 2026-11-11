"""
Interactive Prototype Landing Page
Calibrated Concept-Based Feedback Model (CSE 4385)

Week 1 Status: Project initialization and environment setup.
No models are trained yet. No fabricated predictions or mock metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path so config can be loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Graceful check if Streamlit is not yet installed in the active environment
try:
    import streamlit as st
except ImportError:
    print("\n" + "=" * 65)
    print("Streamlit is not currently installed in this Python environment.")
    print("=" * 65)
    print("To install Streamlit, activate your virtual environment and run:")
    print("    pip install streamlit")
    print("\nThen run this app using:")
    print("    streamlit run app/streamlit_app.py\n")
    sys.exit(0)

from src.config import load_config, ConfigError


def main():
    # Page setup
    st.set_page_config(
        page_title="Calibrated Concept-Based Feedback Model",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header section
    st.title("🦅 Calibrated Concept-Based Feedback Model")
    st.subheader("Interactive Human–AI Correction through Uncertainty-Aware Concept Bottlenecks")
    st.markdown("**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work | **Institution:** Northern University Bangladesh")

    st.divider()

    # Load project configuration
    try:
        config = load_config()
    except ConfigError as e:
        st.error(f"Failed to load project configuration: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("📋 Project Navigation")
        st.info("Current Phase: **Week 1**")
        st.markdown("""
        - [Overview](#project-overview)
        - [Week 1 Development Status](#week-1-development-status)
        - [Configuration Summary](#configuration-summary)
        - [14-Week Roadmap](#14-week-roadmap)
        """)
        st.markdown("---")
        st.caption("Calibrated Concept Feedback Project • CSE 4385")

    # Section 1: Overview
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    Standard Deep Neural Networks classify bird species directly from pixel data ($x \\to y$), functioning as uninterpretable black boxes.
    This project investigates a **Concept Bottleneck Model (CBM)** pipeline ($x \\to c \\to y$) that predicts human-understandable visual attributes (such as *bill shape*, *wing colour*, and *underparts pattern*) before predicting the target bird species.
    
    The research tackles three core challenges:
    1. **Trustworthy Concept Uncertainty:** Applying **temperature scaling** to calibrate raw concept probabilities.
    2. **Principled Review Flagging:** Using **split conformal prediction** to construct set-valued concept predictions $C_j(x)$ and flag non-singleton uncertain concepts for human review.
    3. **Efficient Intervention Policies:** Measuring whether calibrated uncertainty directs a limited human correction budget ($k$ edits) more efficiently than fixed-threshold or random selection.
    """)

    # Section 2: Week 1 Development Status
    st.markdown("### 📌 Week 1 Development Status")
    st.warning("⚠️ **IMPORTANT NOTICE: The model has NOT yet been trained.**")
    st.markdown("""
    In accordance with the 14-week course schedule, **Week 1 focuses strictly on project scaffolding, literature alignment, configuration management, and environment verification**:
    
    - ✅ **No fake predictions:** This interface does not display fabricated species predictions or mocked concept values.
    - ✅ **No simulated metrics:** Calibration errors and accuracy curves will only be reported once models are trained in future weeks.
    - ⏳ **Upcoming Milestones:**
      - **Weeks 3–4:** Train the end-to-end ResNet baseline.
      - **Weeks 5–6:** Train independent, sequential, and joint CBM models.
      - **Week 7:** Fit temperature scaling calibration on validation data.
      - **Week 8:** Calibrate conformal prediction quantiles.
      - **Weeks 9–10:** Execute controlled intervention experiments ($P_1, P_2, P_3$).
      - **Week 12:** Connect trained checkpoints to this interactive Streamlit interface for live human feedback.
    """)

    # Section 3: Configuration Summary
    st.markdown("### ⚙️ Configured Experimental Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Target Bird Species", value=config["data"]["num_classes"])
        st.caption("CUB-200-2011 species classes")

    with col2:
        st.metric(label="Filtered Concepts", value=config["data"]["num_concepts"])
        st.caption("Filtered binary attributes ($c \\in \\{0, 1\\}^{112}$)")

    with col3:
        st.metric(label="Primary Backbone", value=config["model"]["backbone"].upper())
        st.caption(f"Fallback: {config['model']['fallback_backbone'].upper()}")

    st.markdown("#### Grounded Proposal Settings")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**Data Partitions (from 5,994 official train images):**")
        st.markdown(f"- **Training Split:** {config['data']['train_split_ratio']*100:.0f}% (~4,795 images)")
        st.markdown(f"- **Validation Split:** {config['data']['val_split_ratio']*100:.0f}% (~600 images for model selection & temperature scaling)")
        st.markdown(f"- **Calibration Split:** {config['data']['calib_split_ratio']*100:.0f}% (~599 images for conformal quantiles)")
        st.markdown(f"- **Official Test Set:** {config['data']['official_test_count']} images (locked until final evaluation)")

    with exp_col2:
        st.markdown("**Uncertainty & Intervention Policies:**")
        st.markdown(f"- **Calibration Method:** {config['calibration']['method'].replace('_', ' ').title()}")
        st.markdown(f"- **Conformal Significance Levels ($\\alpha$):** {config['conformal']['alpha_levels']}")
        st.markdown(f"- **Intervention Policies:** P1 (Lowest Confidence), P2 (Conformal-Flagged), P3 (Random)")
        st.markdown(f"- **Replacement Strategies:** S1 (Probability), S2 (Percentile Logit)")

    # Section 4: 14-Week Roadmap
    st.markdown("### 🗺️ 14-Week Course Roadmap")
    schedule_data = [
        {"Weeks": "1–2", "Activities": "Literature review, proposal alignment, environment setup, dataset audit.", "Status": "In Progress (Week 1 Complete)"},
        {"Weeks": "3–4", "Activities": "Implement data loaders; train end-to-end ResNet baseline.", "Status": "Planned"},
        {"Weeks": "5–6", "Activities": "Implement independent, sequential, and joint CBMs.", "Status": "Planned"},
        {"Weeks": "7", "Activities": "Implement temperature scaling calibration; evaluate ECE/NLL.", "Status": "Planned"},
        {"Weeks": "8", "Activities": "Split conformal prediction implementation & fixed-threshold comparison.", "Status": "Planned"},
        {"Weeks": "9–10", "Activities": "Intervention experiments (P1/P2/P3), replacement ablations (S1/S2), noisy feedback.", "Status": "Planned"},
        {"Weeks": "11", "Activities": "Statistical aggregation over 3 seeds, error analysis, and figures.", "Status": "Planned"},
        {"Weeks": "12", "Activities": "Connect trained models to this interactive Streamlit UI for user testing.", "Status": "Planned"},
        {"Weeks": "13", "Activities": "Reproducibility audit and final report writing.", "Status": "Planned"},
        {"Weeks": "14", "Activities": "Final revisions and presentation.", "Status": "Planned"},
    ]
    st.table(schedule_data)

    st.markdown("---")
    st.caption("CSE 4385: Artificial Intelligence and Expert System Lab • Northern University Bangladesh")


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        print("\n" + "=" * 65)
        print("Calibrated Concept-Based Feedback Model - Streamlit Application")
        print("=" * 65)
        print("To launch the interactive browser interface, run:")
        print("    .\\venv\\Scripts\\streamlit.exe run app/streamlit_app.py")
        print("or (if venv is activated):")
        print("    streamlit run app/streamlit_app.py\n")

