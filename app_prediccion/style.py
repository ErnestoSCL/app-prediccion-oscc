import streamlit as st

def apply_custom_style():
    """Aplica CSS adaptativo para mejorar la coherencia entre modo claro y oscuro."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Fixes - Coherencia en ambos modos */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    
    /* Forzar visibilidad de textos en el sidebar (blanco sobre fondo oscuro) */
    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    .st-emotion-cache-17l78vu,
    .st-emotion-cache-6q9sum {
        color: #F8FAFC !important;
    }

    /* Radio Button Fix: Círculos y estados activos */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #CBD5E1 !important;
        font-size: 14px;
        font-weight: 500;
    }

    /* Main Area Adaptive Theme */
    /* Evitar el negro puro en modo oscuro, usar un tono suave */
    .main {
        background-color: transparent;
    }

    /* Tarjetas de Métricas Adaptativas */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(226, 232, 240, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }
    
    /* En modo claro la tarjeta debe ser blanca */
    @media (prefers-color-scheme: light) {
        .metric-card {
            background: #FFFFFF;
            border-color: #E2E8F0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
    }

    /* Footer - Sincronizado con el tema */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent; /* No fondo sólido que rompa el tema */
        backdrop-filter: blur(5px);
        color: #64748B;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid rgba(226, 232, 240, 0.1);
        z-index: 1000;
    }

    /* Botones Professional */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    </style>
    """, unsafe_allow_html=True)

def metric_card(title, value, description, color="#38BDF8"):
    """Componente visual para mostrar métricas en tarjetas que funcionan en ambos temas."""
    st.markdown(f'''
    <div class="metric-card">
        <div style="font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        <div style="font-size: 32px; color: {color}; font-weight: 700; margin: 4px 0;">{value}</div>
        <div style="font-size: 13px; color: #64748B; line-height: 1.4;">{description}</div>
    </div>
    ''', unsafe_allow_html=True)

def info_box(content):
    """Caja de información adaptativa para recomendaciones."""
    st.markdown(f'''
    <div style="background-color: rgba(15, 23, 42, 0.05); padding: 20px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-bottom: 25px; border: 1px solid rgba(56, 189, 248, 0.1);">
        <div style="font-weight: 600; color: #38BDF8; margin-bottom: 6px; font-size: 16px;">Recomendaciones Clínicas de Captura</div>
        <div style="font-size: 14px; line-height: 1.6;">{content}</div>
    </div>
    ''', unsafe_allow_html=True)
