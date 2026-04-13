# Diagnóstico Asistido de Carcinoma Oral (OSCC)

Este repositorio contiene una plataforma de patología digital basada en aprendizaje profundo para la asistencia en el diagnóstico del **Carcinoma de Células Escamosas Orales (OSCC)** a partir de micrografías histopatológicas.

## Propósito del Proyecto
El cáncer bucal es una de las neoplasias más prevalentes a nivel global. El objetivo de esta herramienta es proporcionar a los patólogos una segunda opinión basada en datos, identificando regiones críticas mediante mapas de interpretabilidad (**Score-CAM**) y clasificando muestras de tejido con una precisión superior al 97%.

## Estructura del Repositorio
- `app_prediccion/`: Código fuente de la aplicación web profesional (Streamlit).
- `models/`: Pesos del modelo EfficientNet-B0 entrenado.
- `notebooks/`: Flujos de trabajo de entrenamiento y preprocesamiento de imágenes.
- `results/`: Métricas de validación, matriz de confusión e historial de entrenamiento.
- `data_procesada/`: Incluye el `manifiesto.csv` con el catálogo de imágenes utilizadas.

## Tecnologías Utilizadas
- **Lenguaje**: Python 3.10+
- **Deep Learning**: PyTorch & Torchvision (EfficientNet-B0)
- **Interfaz**: Streamlit (Diseño minimalista inspirado en Google Stitch)
- **Visión Artificial**: Score-CAM para interpretabilidad clínica.
- **Gráficos y Datos**: Matplotlib, NumPy y Pandas.

## Arquitectura del Modelo
El sistema utiliza una arquitectura **EfficientNet-B0** como extractor de características dinámico, complementada con una cabeza de clasificación personalizada diseñada específicamente para este tipo de muestras:
1. **Backbone**: EfficientNet-B0 (Compound Scaling).
2. **Dense Head**: Capa de 256 neuronas con activación ReLU.
3. **Regularización**: Batch Normalization y Dropout (30%) para asegurar la generalización del modelo.
4. **Output**: Activación Sigmoide para la clasificación binaria (Normal vs. OSCC).

## Guía de Ejecución Local
Para ejecutar la aplicación en su entorno local, siga estos pasos:

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/ErnestoSCL/app-prediccion-oscc.git
   cd app-prediccion-oscc
   ```

2. Instalar dependencias:
   ```bash
   pip install -r app_prediccion/requirements.txt
   ```

3. Lanzar Streamlit:
   ```bash
   python -m streamlit run app_prediccion/app.py
   ```

## Dataset Original
El modelo fue entrenado con el conjunto de datos: **[Histopathological Imaging Dataset for Oral Cancer](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset/data)**.
**DOI**: 10.17632/ftmp4cvtmb.1

---
*Aviso: Esta herramienta es estrictamente para fines académicos y de apoyo a la investigación médica. No sustituye el juicio clínico de un patólogo certificado.*
