# 🧬 Predicción de Péptidos Antimicrobianos con ProGen2

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Jupyter-Notebook-orange
https://img.shields.io/badge/PyTorch-Deep%2520Learning-red

Repositorio de tesis de maestría en Análítica de Datos - Sistema de predicción de péptidos antimicrobianos mediante fine-tuning del modelo ProGen2.

📋 Tabla de Contenidos
Descripción

Estructura

Modelos

Instalación

Uso

Configuración

Troubleshooting

Resultados

Tecnologías

📖 Descripción
Este proyecto implementa un sistema de predicción de péptidos antimicrobianos utilizando técnicas de fine-tuning sobre el modelo de lenguaje ProGen2. El trabajo forma parte de una tesis de maestría en Análítica de Datos y busca contribuir al descubrimiento de nuevos péptidos terapéuticos.

📁 Estructura del Repositorio
🧪 Modelos
📁 Modelos/ - Contiene todos los Jupyter notebooks de implementación

🧫 Modelo_bac.ipynb - Fine-tuning para péptidos antibacterianos

🍄 Modelo_fungi.ipynb - Fine-tuning para péptidos antifúngicos

🦠 Modelo_viral.ipynb - Fine-tuning para péptidos antivirales

🎗️ Modelo_cancer.ipynb - Fine-tuning para péptidos anticancerígenos

🔬 Modelo_HIV.ipynb - Fine-tuning para péptidos anti-HIV

💊 Modelo_MRSA.ipynb - Fine-tuning para péptidos anti-MRSA

🦠 Modelo_tuber.ipynb - Fine-tuning para péptidos anti-tuberculosis

📊 estadistica_BLAST.ipynb - Análisis estadístico y alineamiento BLAST

🗂️ Datasets
📁 Datasets/ - 7 conjuntos de datos en formato FASTA para entrenamiento

🦠 antibacterial_sequences.fasta (393 KB)

🍄 antifungi_sequences.fasta (157 KB)

🦠 antiviral_sequences.fasta (48 KB)

🎗️ anticancer_sequences.fasta (31 KB)

🔬 antiHIV_sequences.fasta (25 KB)

💊 antiMRSA_sequences.fasta (12 KB)

🦠 antiberculosis_sequences.fasta (0.4 KB)

🐍 Scripts Python
🐍 prepare_data.py - Preparación y preprocesamiento de datos

🐍 finetune.py - Script de fine-tuning de modelos ProGen2

🐍 sample.py - Generación de nuevas secuencias de péptidos

📊 final_data.xlsx - Dataset consolidado para análisis

🧬 Modelos Incluidos
Modelo	Tipo	Archivo
Antibacteriano	Péptidos contra bacterias	Modelos/Modelo_bac.ipynb
Antifúngico	Péptidos contra hongos	Modelos/Modelo_fungi.ipynb
Antiviral	Péptidos contra virus	Modelos/Modelo_viral.ipynb
Anticancer	Péptidos anticancerígenos	Modelos/Modelo_cancer.ipynb
Anti-HIV	Péptidos específicos HIV	Modelos/Modelo_HIV.ipynb
Anti-MRSA	Péptidos contra MRSA	Modelos/Modelo_MRSA.ipynb
Anti-Tuberculosis	Péptidos contra TB	Modelos/Modelo_tuber.ipynb
⚙️ Instalación
# Clonar el repositorio
git clone https://github.com/plermab/Maestria-analitica-de-Datos---Tesis-Prediccion-de-peptidos-antimicrobianos.git
cd Maestria-analitica-de-Datos---Tesis-Prediccion-de-peptidos-antimicrobianos

# Instalar dependencias
pip install torch transformers pandas numpy jupyter scikit-learn biopython

🛠️ Configuración del Entorno
Requisitos del Sistema
GPU: NVIDIA con ≥ 8GB VRAM (recomendado)

RAM: ≥ 16GB

Almacenamiento: ≥ 10GB para modelos y datasets

Instalación Completa