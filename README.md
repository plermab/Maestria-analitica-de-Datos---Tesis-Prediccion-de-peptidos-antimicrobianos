# Predicción de Péptidos Antimicrobianos

Este repositorio contiene los modelos y scripts para la predicción de péptidos antimicrobianos utilizando fine-tuning de ProGen2, desarrollado como parte de mi trabajo de tesis de maestría.

## 📁 Estructura del repositorio

- `*.ipynb` - Notebooks de Jupyter con los modelos de predicción
- `prepare_data.py` - Script para preparación de datos
- `finetune.py` - Script para fine-tuning de modelos
- `sample.py` - Script para generación de muestras
- `final_data.xlsx` - Dataset final de péptidos

## 🧬 Modelos incluidos

- **Modelo_bac.ipynb** - Predicción de péptidos antibacterianos
- **Modelo_fungi.ipynb** - Predicción de péptidos antifúngicos  
- **Modelo_viral.ipynb** - Predicción de péptidos antivirales
- **Modelo_cancer.ipynb** - Predicción de péptidos anticancerígenos
- **Modelo_HIV.ipynb** - Predicción de péptidos anti-HIV
- **Modelo_MRSA.ipynb** - Predicción de péptidos anti-MRSA
- **Modelo_tuber.ipynb** - Predicción de péptidos anti-tuberculosis

## 🚀 Uso

### Preparación de datos
```bash
python prepare_data.py
###Fine-tuning
python finetune.py
###Generación de muestras
python sample.py

