import io
import os
from pathlib import Path
import mimetypes

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image

from inference import HistopathologyPredictor
import style


def load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env_file()
API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="Oral Cancer Diagnostics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

style.apply_custom_style()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "best_model_efficientnet_variante.pth"
SAMPLE_DIR = BASE_DIR / "assets" / "samples"


@st.cache_resource(show_spinner=False)
def get_predictor() -> HistopathologyPredictor:
    return HistopathologyPredictor(str(MODEL_PATH))


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def guess_content_type(filename: str | None) -> str:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed and guessed.startswith("image/"):
            return guessed
    return "image/png"


def predict_with_api(image_bytes: bytes, filename: str = "histopathology.png", content_type: str = "image/png") -> dict:
    if not API_URL:
        raise RuntimeError("API_URL no está configurada")

    files = {
        "file": (filename, image_bytes, content_type),
    }
    endpoint = f"{API_URL.rstrip('/')}/predict"
    response = requests.post(endpoint, files=files, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"La API respondió {response.status_code}: {response.text}")
    return response.json()


def get_prediction(
    predictor: HistopathologyPredictor | None,
    image: Image.Image,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict:
    if API_URL:
        return predict_with_api(image_bytes=image_bytes, filename=filename, content_type=content_type)
    if predictor is None:
        raise RuntimeError("Predictor local no inicializado")
    return predictor.predict(image)


with st.sidebar:
    st.markdown("### Navegación")
    page = st.radio(
        "Seleccione una sección:",
        ["Diagnóstico Asistido", "Rendimiento del Modelo", "Centro de Información"],
    )
    st.markdown("---")
    if API_URL:
        st.caption(f"Modo API activo: `{API_URL}`")
    else:
        st.caption("Modo local activo: inferencia en Streamlit")


if page == "Diagnóstico Asistido":
    st.title("Diagnóstico de Carcinoma de Células Escamosas Orales asistido por IA")

    style.info_box(
        """
    <strong>Protocolo de Calidad de Imagen:</strong><br>
    • Utilice muestras teñidas con Hematoxilina y Eosina (H&E).<br>
    • Asegure un aumento digital entre 100x y 400x para visualizar la arquitectura celular.<br>
    • Evite áreas con excesivos artefactos de fijación o burbujas.<br>
    • El archivo debe estar en formato JPEG o PNG con iluminación compensada.
    """
    )

    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("Carga de Muestra")
        uploaded_file = st.file_uploader(
            "Subir micrografia histopatologica...", type=["jpg", "png", "jpeg"]
        )

        st.markdown("<br>Galería de Pruebas Rápidas", unsafe_allow_html=True)
        samples = sorted(os.listdir(SAMPLE_DIR)) if SAMPLE_DIR.exists() else []
        selected_sample = None

        if samples:
            cols = st.columns(len(samples))
            for i, sample_name in enumerate(samples):
                with cols[i]:
                    img_path = SAMPLE_DIR / sample_name
                    label = "OSCC" if "OSCC" in sample_name else "Normal"
                    st.image(str(img_path), use_container_width=True)
                    if st.button(f"Test {label}", key=f"btn_{i}"):
                        selected_sample = img_path

    image_to_process = None
    image_bytes = b""
    image_filename = "histopathology.png"
    image_content_type = "image/png"

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_filename = uploaded_file.name or "uploaded_image.png"
        image_content_type = uploaded_file.type or guess_content_type(image_filename)
        image_to_process = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif selected_sample is not None:
        image_bytes = selected_sample.read_bytes()
        image_filename = selected_sample.name
        image_content_type = guess_content_type(image_filename)
        image_to_process = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with col_result:
        st.subheader("Resultados de Clasificación")
        if image_to_process:
            predictor = get_predictor()

            with st.spinner("Analizando morfológicamente la muestra..."):
                result = get_prediction(
                    predictor,
                    image_to_process,
                    image_bytes=image_bytes,
                    filename=image_filename,
                    content_type=image_content_type,
                )
                probability_oscc = float(result.get("probability_oscc", 0.0))
                confidence = float(result.get("confidence", 0.0))
                prediction = str(result.get("label", "Resultado desconocido"))
                is_oscc = probability_oscc > 0.5
                color = "#B91C1C" if is_oscc else "#15803D"

                cam_map, _ = predictor.score_cam(image_to_process)

            st.markdown(
                f"""
            <div style="background-color: {color}10; padding: 24px; border-radius: 12px; border: 2px solid {color}; margin-bottom: 25px;">
                <div style="font-size: 14px; color: {color}; font-weight: 600; text-transform: uppercase;">Estado Diagnosticado</div>
                <h2 style="color: {color}; margin: 5px 0 0 0;">{prediction}</h2>
                <div style="margin-top: 10px; font-size: 16px; color: #334155;">Confianza del Modelo: <strong>{confidence:.2%}</strong></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Interpretación Visual (Score-CAM)")
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plt.subplots_adjust(wspace=0.1)

            ax[0].imshow(image_to_process)
            ax[0].set_title("Original", fontsize=10, pad=10)
            ax[0].axis("off")

            cam_pil = Image.fromarray((cam_map * 255).astype(np.uint8))
            cam_map_resized = np.array(cam_pil.resize((224, 224), Image.BILINEAR)) / 255.0
            heatmap = plt.cm.jet(cam_map_resized)[:, :, :3]
            original_np = np.array(image_to_process.resize((224, 224))) / 255.0
            overlay = 0.6 * original_np + 0.4 * heatmap

            ax[1].imshow(overlay)
            ax[1].set_title("Regiones de Interés Clínico", fontsize=10, pad=10)
            ax[1].axis("off")

            st.pyplot(fig)
            st.caption(
                "Nota: Las zonas cálidas (rojas/amarillas) representan las regiones celulares con mayor impacto en el diagnóstico final."
            )
        else:
            st.markdown(
                """
            <div style="height: 350px; display: flex; align-items: center; justify-content: center; background: #F8FAFC; border: 2px dashed #E2E8F0; border-radius: 12px; color: #64748B;">
                Esperando carga de datos para iniciar el análisis biomédico
            </div>
            """,
                unsafe_allow_html=True,
            )

elif page == "Rendimiento del Modelo":
    st.title("Validación Científica y Rendimiento")
    st.markdown("Resultados obtenidos en un conjunto de prueba independiente de 520 micrografías.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        style.metric_card("Accuracy", "97.31%", "Exactitud global del sistema")
    with m2:
        style.metric_card("F1-Score", "0.974", "Armonía entre Precisión y Recall", color="#0369A1")
    with m3:
        style.metric_card("AUC-ROC", "0.995", "Capacidad discriminativa del modelo", color="#15803D")
    with m4:
        style.metric_card("Falsos Negativos", "8 casos", "Tasa crítica para oncología clínica", color="#B91C1C")

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2, gap="large")

    with col_chart1:
        st.markdown("#### Historial de Convergencia")
        train_acc = [0.934, 0.935, 0.939, 0.946, 0.948, 0.946, 0.952, 0.952, 0.949, 0.955, 0.956, 0.957, 0.958, 0.960, 0.960, 0.965, 0.961, 0.961, 0.971, 0.969]
        val_acc = [0.927, 0.931, 0.927, 0.931, 0.934, 0.929, 0.933, 0.925, 0.940, 0.936, 0.936, 0.940, 0.933, 0.944, 0.944, 0.942, 0.938, 0.940, 0.942, 0.944]

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(train_acc, label="Training Accuracy", color="#0F172A", linewidth=2.5)
        ax2.plot(val_acc, label="Validation Accuracy", color="#38BDF8", linewidth=2.5, linestyle="--")
        ax2.set_xlabel("Evolución por Épocas", fontsize=9, color="#64748B")
        ax2.set_ylabel("Precisión", fontsize=9, color="#64748B")
        ax2.legend(frameon=False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", linestyle=":", alpha=0.5)
        st.pyplot(fig2)

    with col_chart2:
        st.markdown("#### Matriz de Confusión")
        cm = np.array([[244, 6], [8, 262]])
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        ax3.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, alpha=0.8)

        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["Normal", "OSCC"], fontsize=9)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["Normal", "OSCC"], fontsize=9)

        for i in range(2):
            for j in range(2):
                ax3.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > 150 else "#0F172A",
                    fontsize=12,
                    fontweight="bold",
                )

        ax3.set_ylabel("Realidad Clínica", fontsize=9, color="#64748B")
        ax3.set_xlabel("Predicción Computacional", fontsize=9, color="#64748B")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax3.spines["bottom"].set_visible(False)
        st.pyplot(fig3)

elif page == "Centro de Información":
    st.title("Documentación Técnica y Repositorio")

    col_text, col_tech = st.columns([1.5, 1], gap="large")

    with col_text:
        st.markdown(
            """
        ### El Cáncer Bucal y el Carcinoma de Células Escamosas (OSCC)
        El Carcinoma de Células Escamosas Orales es la neoplasia maligna más frecuente de la cavidad oral. Este proyecto nace como una herramienta de apoyo para patólogos, facilitando la detección precoz mediante el análisis automatizado de patrones de queratinización, pleomorfismo nuclear e invasión estromal en micrografías digitales.

        ### Dataset y Referencia Científica
        Este sistema ha sido validado utilizando el **Histopathological Imaging Dataset for Oral Cancer**, un repositorio de acceso abierto que contiene muestras pareadas de epitelio normal y carcinoma.

        - **Fuente de Datos**: [Kaggle - Histopathological Imaging Dataset for Oral Cancer](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset/data)
        - **Cita DOI**: [10.17632/ftmp4cvtmb.1](https://doi.org/10.17632/ftmp4cvtmb.1)
        - **Título Original**: *A histopathological image repository of normal epithelium of Oral Cavity and Oral Squamous Cell Carcinoma*.

        ### Repositorio del Proyecto
        El código fuente, los procedimientos de aumentación de datos y el pipeline de entrenamiento están disponibles en:

        [github.com/ErnestoSCL/app-prediccion-oscc](https://github.com/ErnestoSCL/app-prediccion-oscc)

        *Nota: Los pesos de los modelos VGG16 no se incluyen en el repositorio debido a su gran tamaño (>500MB).*
        """
        )

    with col_tech:
        st.markdown("### Arquitectura del Modelo")
        st.info(
            """
        El diseño final se fundamenta en un modelo de transferencia de aprendizaje optimizado específicamente para este dataset:

        1. **Extractor de Características:** Backbone **EfficientNet-B0**, seleccionado por su escalado compuesto de profundidad, ancho y resolución, ideal para detectar texturas histológicas microfinas.
        2. **Cabeza de Clasificación (Custom):**
            - Capa Linear (input: 1280, output: 256).
            - **Batch Normalization (1D):** Implementada para estabilizar la distribución de activaciones y acelerar la convergencia.
            - Activación ReLU.
            - **Dropout (30%):** Tasa de abandono optimizada para prevenir el sobreajuste a patrones de ruido cromático en las tinciones.
            - Capa Linear final (256 a 1) seguida de activación sigmoide para probabilidad binaria.
        """
        )

st.markdown(
    """
<div class="footer">
    Plataforma de Investigación en Patología Digital. Uso estrictamente académico y de soporte diagnóstico.
</div>
""",
    unsafe_allow_html=True,
)
