# 🧬 Predicción de Péptidos Antimicrobianos con ProGen2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org)

Repositorio de tesis de maestría en Análítica de Datos - Sistema de predicción de péptidos antimicrobianos mediante fine-tuning del modelo ProGen2.

## 📋 Tabla de Contenidos
- [Descripción](#descripción)
- [Estructura](#estructura-del-repositorio)
- [Modelos](#-modelos-incluidos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Tecnologías](#-tecnologías)

## 📖 Descripción

Este proyecto implementa un sistema de predicción de péptidos antimicrobianos utilizando técnicas de fine-tuning sobre el modelo de lenguaje ProGen2. El trabajo forma parte de una tesis de maestría en Análítica de Datos y busca contribuir al descubrimiento de nuevos péptidos terapéuticos.

## 📁 Estructura del Repositorio
<<<<<<< HEAD
## 📁 Estructura del Repositorio

- **Modelo_bac.ipynb** - Modelo para péptidos antibacterianos  
- **Modelo_fungi.ipynb** - Modelo para péptidos antifúngicos  
- **Modelo_viral.ipynb** - Modelo para péptidos antivirales  
- **Modelo_cancer.ipynb** - Modelo para péptidos anticancerígenos  
- **Modelo_HIV.ipynb** - Modelo para péptidos anti-HIV  
- **Modelo_MRSA.ipynb** - Modelo para péptidos anti-MRSA  
- **Modelo_tuber.ipynb** - Modelo para péptidos anti-tuberculosis  
- **estadistica_BLAST.ipynb** - Análisis estadístico y BLAST  
- **prepare_data.py** - Preparación y preprocesamiento de datos  
- **finetune.py** - Script de fine-tuning de modelos  
- **sample.py** - Generación de nuevas secuencias  
- **final_data.xlsx** - Dataset completo de péptidos  
- **README.md** - Este archivo
=======
-📓 Modelo_bac.ipynb # Modelo para péptidos antibacterianos.
-📓 Modelo_fungi.ipynb # Modelo para péptidos antifúngicos.
-📓 Modelo_viral.ipynb # Modelo para péptidos antivirales.
-📓 Modelo_cancer.ipynb # Modelo para péptidos anticancerígenos.
-📓 Modelo_HIV.ipynb # Modelo para péptidos anti-HIV.
-📓 Modelo_MRSA.ipynb # Modelo para péptidos anti-MRSA.
-📓 Modelo_tuber.ipynb # Modelo para péptidos anti-tuberculosis.
-📊 estadistica_BLAST.ipynb # Análisis estadístico y BLAST.
-🐍 prepare_data.py # Preparación y preprocesamiento de datos.
-🐍 finetune.py # Script de fine-tuning de modelos.
-🐍 sample.py # Generación de nuevas secuencias.
-📈 final_data.xlsx # Dataset completo de péptidos.
-📄 README.md # Este archivo.
>>>>>>> f55d42e43ef411a1c8042dd27a410e4236a58005

## 🧬 Modelos Incluidos

| Modelo | Tipo | Archivo |
|--------|------|---------|
| **Antibacteriano** | Péptidos contra bacterias | `Modelo_bac.ipynb` |
| **Antifúngico** | Péptidos contra hongos | `Modelo_fungi.ipynb` |
| **Antiviral** | Péptidos contra virus | `Modelo_viral.ipynb` |
| **Anticancer** | Péptidos anticancerígenos | `Modelo_cancer.ipynb` |
| **Anti-HIV** | Péptidos específicos HIV | `Modelo_HIV.ipynb` |
| **Anti-MRSA** | Péptidos contra MRSA | `Modelo_MRSA.ipynb` |
| **Anti-Tuberculosis** | Péptidos contra TB | `Modelo_tuber.ipynb` |

## ⚙️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/plermab/Maestria-analitica-de-Datos---Tesis-Prediccion-de-peptidos-antimicrobianos.git
cd Maestria-analitica-de-Datos---Tesis-Prediccion-de-peptidos-antimicrobianos

# Instalar dependencias (ejemplo)
pip install torch transformers pandas numpy jupyter
🚀 Uso
1. Preparación de Datos
bash
python prepare_data.py
2. Entrenamiento (Fine-tuning)
bash
python finetune.py
3. Generación de Muestras
bash
python sample.py
4. Análisis en Jupyter
bash
jupyter notebook
# Abrir cualquiera de los notebooks de modelo_*.ipynb
📊 Resultados
Los modelos fueron evaluados utilizando métricas de:

Precisión en la predicción de actividad antimicrobiana

Diversidad de secuencias generadas

Similitud con péptidos naturales

Potencial terapéutico estimado

🔬 Tecnologías
Python 3.8+ - Lenguaje principal

PyTorch - Framework de deep learning

Transformers - Fine-tuning de ProGen2

Jupyter Notebook - Análisis y visualización

Pandas/Numpy - Procesamiento de datos

ProGen2 - Modelo base para fine-tuning

📝 Cita este Trabajo
Si utilizas este código en tu investigación, por favor cita:

text
Tesis de Maestría en Análítica de Datos - Predicción de Péptidos Antimicrobianos
Autor: Paula Andrea Lerma Barbosa
Año: 2025
