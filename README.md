# Prediccion Asistida de Carcinoma Oral (OSCC)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)

Aplicacion de patologia digital para apoyo en la prediccion de **Carcinoma de Celulas Escamosas Orales (OSCC)** a partir de micrografias histopatologicas, con salida probabilistica e interpretabilidad visual mediante Score-CAM.

## Demo en vivo

Puedes probar la aplicacion desplegada en Streamlit aqui:

- https://app-prediccion-oscc.streamlit.app/

## Objetivo del proyecto

Este repositorio busca apoyar el tamizaje y priorizacion de casos de cancer bucal mediante vision por computadora, ofreciendo:

- Clasificacion binaria (Normal vs OSCC).
- Visualizacion de regiones de interes clinico (Score-CAM).
- Interfaz web para evaluacion rapida por imagen.

Aviso: herramienta de apoyo academico e investigativo. No reemplaza diagnostico de un profesional de patologia.

## Arquitectura del modelo

El sistema utiliza un backbone CNN con ajuste para clasificacion binaria:

1. Backbone EfficientNet-B0 para extraccion de caracteristicas.
2. Cabeza densa personalizada con 256 neuronas y activacion ReLU.
3. Regularizacion con BatchNorm y Dropout (30%).
4. Capa de salida sigmoide para estimacion de probabilidad de OSCC.

## Estructura del repositorio

```text
app-prediccion-oscc/
├── README.md
├── app_prediccion/
│   ├── app.py
│   ├── model_utils.py
│   ├── style.py
│   ├── requirements.txt
│   ├── assets/
│   └── models/
├── models/
├── notebooks/
├── results/
└── data_procesada/
```

## Tecnologias utilizadas

- Python 3.10+
- PyTorch y Torchvision
- Streamlit
- NumPy, Pandas y Matplotlib
- Score-CAM para interpretabilidad

## Ejecucion local

1. Clonar el repositorio:

```bash
git clone https://github.com/ErnestoSCL/app-prediccion-oscc.git
cd app-prediccion-oscc
```

2. Instalar dependencias:

```bash
pip install -r app_prediccion/requirements.txt
```

3. Ejecutar la aplicacion:

```bash
python -m streamlit run app_prediccion/app.py
```

Por defecto, Streamlit abre la app en `http://localhost:8501`.

## Datos y referencia

Dataset base utilizado en entrenamiento:

- Histopathological Imaging Dataset for Oral Cancer
- https://www.kaggle.com/datasets/ashenafifasilkebede/dataset/data
- DOI: 10.17632/ftmp4cvtmb.1

## Licencia

El proyecto se distribuye bajo licencia MIT.