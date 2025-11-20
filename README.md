# 🧬 Predicción de Péptidos Antimicrobianos con ProGen2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org)

Repositorio de tesis de maestría en Análítica de Datos - Sistema de predicción de péptidos antimicrobianos mediante fine-tuning del modelo ProGen2.

## 📋 Tabla de Contenidos
- [Descripción](#descripción)
- [Estructura](#estructura-del-repositorio)
- [Modelos](#modelos-incluidos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Configuración](#configuración-del-entorno)
- [Troubleshooting](#troubleshooting)
- [Resultados](#resultados)
- [Tecnologías](#tecnologías)

## 📖 Descripción

Este proyecto implementa un sistema de predicción de péptidos antimicrobianos utilizando técnicas de fine-tuning sobre el modelo de lenguaje ProGen2. El trabajo forma parte de una tesis de maestría en Análítica de Datos y busca contribuir al descubrimiento de nuevos péptidos terapéuticos.

## 📁 Estructura del Repositorio

### 🧪 Modelos
- **📁 Modelos/** - Contiene todos los Jupyter notebooks de implementación
  - **🧫 Modelo_bac.ipynb** - Fine-tuning para péptidos antibacterianos  
  - **🍄 Modelo_fungi.ipynb** - Fine-tuning para péptidos antifúngicos  
  - **🦠 Modelo_viral.ipynb** - Fine-tuning para péptidos antivirales  
  - **🎗️ Modelo_cancer.ipynb** - Fine-tuning para péptidos anticancerígenos  
  - **🔬 Modelo_HIV.ipynb** - Fine-tuning para péptidos anti-HIV  
  - **💊 Modelo_MRSA.ipynb** - Fine-tuning para péptidos anti-MRSA  
  - **🦠 Modelo_tuber.ipynb** - Fine-tuning para péptidos anti-tuberculosis  
  - **📊 estadistica_BLAST.ipynb** - Análisis estadístico y alineamiento BLAST

### 🗂️ Datasets
- **📁 Datasets/** - 7 conjuntos de datos en formato FASTA para entrenamiento
  - 🦠 `antibacterial_sequences.fasta` (393 KB)
  - 🍄 `antifungi_sequences.fasta` (157 KB)
  - 🦠 `antiviral_sequences.fasta` (48 KB)
  - 🎗️ `anticancer_sequences.fasta` (31 KB)
  - 🔬 `antiHIV_sequences.fasta` (25 KB)
  - 💊 `antiMRSA_sequences.fasta` (12 KB)
  - 🦠 `antiberculosis_sequences.fasta` (0.4 KB)

### 🐍 Scripts Python
- **🐍 prepare_data.py** - Preparación y preprocesamiento de datos  
- **🐍 finetune.py** - Script de fine-tuning de modelos ProGen2
- **🐍 sample.py** - Generación de nuevas secuencias de péptidos
- **📊 final_data.xlsx** - Dataset consolidado para análisis

## 🧬 Modelos Incluidos

| Modelo | Tipo | Archivo |
|--------|------|---------|
| **Antibacteriano** | Péptidos contra bacterias | `Modelos/Modelo_bac.ipynb` |
| **Antifúngico** | Péptidos contra hongos | `Modelos/Modelo_fungi.ipynb` |
| **Antiviral** | Péptidos contra virus | `Modelos/Modelo_viral.ipynb` |
| **Anticancer** | Péptidos anticancerígenos | `Modelos/Modelo_cancer.ipynb` |
| **Anti-HIV** | Péptidos específicos HIV | `Modelos/Modelo_HIV.ipynb` |
| **Anti-MRSA** | Péptidos contra MRSA | `Modelos/Modelo_MRSA.ipynb` |
| **Anti-Tuberculosis** | Péptidos contra TB | `Modelos/Modelo_tuber.ipynb` |

## ⚙️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/plermab/Maestria-analitica-de-Datos---Tesis-Prediccion-de-peptidos-antimicrobianos.git
cd Maestria-analitica-de-Datos---Tesis-Prediccion-de-peptidos-antimicrobianos

# Instalar dependencias
pip install torch transformers pandas numpy jupyter scikit-learn biopython
``` 
## 🚀 Uso

Este repositorio implementa un flujo de trabajo completo para fine-tuning del modelo ProGen2 en péptidos antimicrobianos.

### 📥 Descarga de Datos
Los datasets ya están incluidos en la carpeta `Datasets/` con 7 tipos de péptidos antimicrobianos en formato FASTA.

### 🔄 Preprocesamiento de Datos
Antes del fine-tuning, preprocesamos los datos para incluir tokens especiales y preparar las secuencias:

```bash
# Preprocesar datos antibacterianos (ejemplo)
python prepare_data.py \
    --input_files Datasets/antibacterial_sequences.fasta \
    --output_file_train=train_antibacterial.txt \
    --output_file_test=test_antibacterial.txt \
    --train_split_ratio=0.8 \
    --bidirectional
```
**Parámetros:**
- `--input_files`: Archivos FASTA de entrada
- `--output_file_train`: Archivo de salida para datos de entrenamiento
- `--output_file_test`: Archivo de salida para datos de prueba
- `--train_split_ratio`: Proporción train/test (default: 0.8)
- `--bidirectional`: Incluir secuencias en reversa para modelo bidireccional

### 🎯 Fine-tuning
Entrena el modelo en los datos preprocesados (se recomienda usar GPU):

```bash
# Fine-tuning para péptidos antibacterianos
python finetune.py \
    --model=hugohrban/progen2-small \
    --train_file=train_antibacterial.txt \
    --test_file=test_antibacterial.txt \
    --device=cuda \
    --epochs=15 \
    --batch_size=16 \
    --accumulation_steps=4 \
    --lr=1e-5 \
    --decay=linear \
    --warmup_steps=200 \
    --eval_before_train
```
**Parámetros principales:**
- `--model`: Modelo base de ProGen2
- `--train_file`: Archivo con datos de entrenamiento
- `--test_file`: Archivo con datos de prueba
- `--device`: Dispositivo (`cuda` o `cpu`)
- `--epochs`: Número de épocas de entrenamiento
- `--batch_size`: Tamaño del batch
- `--lr`: Tasa de aprendizaje
- `--eval_before_train`: Evaluar antes de comenzar el entrenamiento

### 🧬 Generación de Secuencias
Genera nuevas secuencias de péptidos usando el modelo fine-tuned:

```bash
# Generar péptidos antibacterianos
python sample.py \
    --model=checkpoints/progen2-small-best \
    --device=cuda \
    --batch_size=10000 \
    --prompt="1" \
    --iters=1 \
    --min_length=12 \
    --max_length=33 \
    --k=30 \
    --t=1.0
```
**Parámetros de generación:**
- `--model`: Modelo fine-tuned para usar
- `--prompt`: Token de inicio (generalmente "1")
- `--min_length`: Longitud mínima de secuencia
- `--max_length`: Longitud máxima de secuencia
- `--k`: Top-k para sampling
- `--t`: Temperatura para sampling
- `--batch_size`: Número de secuencias a generar

### 📊 Notebooks Específicos
Para cada tipo de péptido, usa los notebooks correspondientes en la carpeta `Modelos/`:

```bash
jupyter notebook Modelos/Modelo_bac.ipynb        # Antibacterianos
jupyter notebook Modelos/Modelo_fungi.ipynb      # Antifúngicos
jupyter notebook Modelos/Modelo_viral.ipynb      # Antivirales
# ... y así para cada tipo
```
## ⚙️ Configuración de Hyperparámetros

### Recomendaciones para Péptidos Cortos
```python
# Para péptidos antimicrobianos (12-50 aa)
config = {
    "batch_size": 16,
    "accumulation_steps": 4,
    "learning_rate": 1e-5,
    "epochs": 15,
    "warmup_steps": 200,
    "max_length": 512
}
```
### Optimización para Recursos Limitados
```bash
# Si tienes memoria GPU limitada
python finetune.py \
    --batch_size=8 \
    --accumulation_steps=8 \
    --lr=5e-6 \
    --epochs=20
```
## 📝 Cita este Trabajo

Si utilizas este código en tu investigación, por favor cita: Tesis de Maestría en Análítica de Datos - Predicción de Péptidos Antimicrobianos
Autor: Paula Andrea Lerma Barbosa
Año: 2025
