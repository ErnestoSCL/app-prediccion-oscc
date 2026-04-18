import streamlit as st


def apply_custom_style() -> None:
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(226, 232, 240, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        backdrop-filter: blur(4px);
        color: #64748B;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
        border-top: 1px solid rgba(226, 232, 240, 0.1);
        z-index: 1000;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, description: str, color: str = "#38BDF8") -> None:
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="font-size:12px;color:#94A3B8;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">{title}</div>
        <div style="font-size:32px;color:{color};font-weight:700;margin:4px 0;">{value}</div>
        <div style="font-size:13px;color:#64748B;line-height:1.4;">{description}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def info_box(content: str) -> None:
    st.markdown(
        f"""
    <div style="background-color: rgba(15, 23, 42, 0.05); padding: 20px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-bottom: 25px; border: 1px solid rgba(56, 189, 248, 0.1);">
        <div style="font-weight: 600; color: #38BDF8; margin-bottom: 6px; font-size: 16px;">Recomendaciones Clínicas de Captura</div>
        <div style="font-size: 14px; line-height: 1.6;">{content}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
