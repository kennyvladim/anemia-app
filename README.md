# 🩸 Predicción de Anemia Infantil — ENDES 2024 (Perú)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Aplicación web para predecir el **riesgo de anemia** en niños menores de 5 años en Perú, basada en datos de la **ENDES 2024 (INEI)**. Modelo Random Forest con 31 predictores socioeconómicos, nutricionales y de salud.

🔗 **Demo en vivo:** [anemia-app.streamlit.app](https://anemia-app-h8f3pdanvzez8ykwr2rv6v.streamlit.app)

---

## 📌 Índice

1. [Contexto del problema](#contexto-del-problema)
2. [Dataset](#dataset)
3. [Metodología](#metodología)
4. [Resultados del modelo](#resultados-del-modelo)
5. [Estructura del repositorio](#estructura-del-repositorio)
6. [Instalación local](#instalación-local)
7. [Predictores](#predictores)
8. [Tecnologías](#tecnologías)
9. [Referencias](#referencias)

---

## 📍 Contexto del problema

La anemia infantil es un problema de salud pública crítico en Perú: según la ENDES 2024, afecta al **43.6% de niños menores de 5 años**, con mayor prevalencia en zonas rurales y de alta altitud. La detección temprana mediante modelos predictivos permite focalizar intervenciones de salud pública antes de que se realice un hemograma.

Este proyecto desarrolla un modelo de machine learning que, a partir de variables del entorno del niño (hogar, madre, gestación, suplementación), estima la probabilidad de anemia sin necesidad de análisis de laboratorio.

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| **Fuente** | ENDES 2024 — Instituto Nacional de Estadística e Informática (INEI) |
| **Cobertura** | Nacional, Perú |
| **Población objetivo** | Niños de 0 a 59 meses |
| **Variable respuesta** | Anemia (Hb < 11.0 g/dL ajustada por altitud) |
| **Predictores** | 31 variables socioeconómicas, nutricionales y de salud |

---

## 🔬 Metodología

```
ENDES 2024 (raw)
      │
      ▼
Limpieza y preprocesamiento
  - Imputación por mediana (SimpleImputer)
  - Estandarización (StandardScaler)
  - Codificación de variables categóricas
      │
      ▼
Selección de variables
  - Revisión bibliográfica (Yimer et al., 2025)
  - Análisis de importancia (feature importance RF)
      │
      ▼
Modelado — Random Forest Classifier
  - Validación cruzada estratificada (5-fold)
  - Ajuste de hiperparámetros (GridSearchCV)
  - class_weight='balanced' (manejo de desbalance)
      │
      ▼
Evaluación → AUC-ROC ~72–75%
      │
      ▼
Despliegue — Streamlit Cloud
```

### Hiperparámetros del modelo final

| Parámetro | Valor |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 12 |
| `min_samples_leaf` | 10 |
| `class_weight` | balanced |
| `random_state` | 42 |

---

## 📈 Resultados del modelo

| Métrica | Valor |
|---|---|
| **AUC-ROC** | ~0.72 – 0.75 |
| **Umbral de clasificación** | 0.50 |
| **Técnica de balanceo** | class_weight balanced |

> ⚠️ El modelo es de carácter investigativo/educativo. No reemplaza el diagnóstico clínico profesional ni el hemograma.

---

## 🗂️ Estructura del repositorio

```
anemia-app/
│
├── app.py                  # Aplicación Streamlit (UI + lógica de predicción)
├── modelo_anemia_rf.pkl    # Modelo serializado con joblib (Pipeline RF)
├── predictores.json        # Orden exacto de los 31 predictores
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

---

## 💻 Instalación local

### Requisitos previos
- Python 3.10 o superior
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/kennyvladim/anemia-app.git
cd anemia-app

# 2. Crear entorno virtual
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Correr la aplicación
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

---

## 📋 Predictores

Los 31 predictores están organizados en 7 bloques temáticos:

| # | Variable | Descripción | Tipo |
|---|---|---|---|
| 1 | `edad_meses` | Edad del niño en meses (0–59) | Numérica |
| 2 | `sexo_nino` | Sexo (1=Masculino, 0=Femenino) | Binaria |
| 3 | `stunting` | Desnutrición crónica (z-talla < −2) | Binaria |
| 4 | `wasting` | Desnutrición aguda (z-peso/talla < −2) | Binaria |
| 5 | `altitud_msnm` | Altitud del hogar (msnm) | Numérica |
| 6 | `area_urbana` | Área de residencia (1=Urbano, 0=Rural) | Binaria |
| 7 | `region` | Código de región ENDES (1–25) | Numérica |
| 8 | `nro_ninos_h` | N.° de niños <5 años en el hogar | Numérica |
| 9 | `indice_riqueza` | Índice de riqueza (1=Muy pobre … 5=Muy rico) | Ordinal |
| 10 | `agua_mejorada` | Acceso a agua mejorada | Binaria |
| 11 | `san_mejorado` | Saneamiento mejorado | Binaria |
| 12 | `comb_limpio` | Uso de combustible limpio | Binaria |
| 13 | `educ_madre` | Nivel educativo de la madre | Ordinal |
| 14 | `madre_trabaja` | Madre trabaja actualmente | Binaria |
| 15 | `madre_sis` | Madre afiliada al SIS | Binaria |
| 16 | `edad_madre` | Edad de la madre (años) | Numérica |
| 17 | `hb_materna` | Hemoglobina materna (g/dL × 10) | Numérica |
| 18 | `ctrl_prenatal_temprano` | Control prenatal en el 1.er trimestre | Binaria |
| 19 | `sin_ctrl_prenatal` | Sin ningún control prenatal | Binaria |
| 20 | `hierro_emb` | Suplemento de hierro durante el embarazo | Binaria |
| 21 | `bajo_peso_nacer` | Bajo peso al nacer | Binaria |
| 22 | `diarrea` | Diarrea en las últimas 2 semanas | Binaria |
| 23 | `tos` | Tos en las últimas 2 semanas | Binaria |
| 24 | `ira` | IRA en las últimas 2 semanas | Binaria |
| 25 | `vitamina_a` | Recibió vitamina A | Binaria |
| 26 | `antiparasit` | Recibió antiparasitario | Binaria |
| 27 | `sis_nino` | Niño afiliado al SIS | Binaria |
| 28 | `hierro_7d` | Hierro jarabe en los últimos 7 días | Binaria |
| 29 | `micronut_minsa` | Micronutrientes MINSA en los últimos 7 días | Binaria |
| 30 | `cred` | Controles CRED al día | Binaria |
| 31 | `hb_materna_falta` | Hemoglobina materna no disponible | Binaria |

---

## 🛠️ Tecnologías

| Herramienta | Uso |
|---|---|
| Python 3.10 | Lenguaje principal |
| scikit-learn 1.6.1 | Modelado (Random Forest, Pipeline, GridSearchCV) |
| pandas / numpy | Procesamiento de datos |
| joblib | Serialización del modelo |
| Streamlit | Interfaz web interactiva |
| Streamlit Cloud | Despliegue gratuito |

---

## 📚 Referencias

- Yimer, T. et al. (2025). *Machine learning-based prediction of anemia in children under five years: Evidence from Ethiopian demographic and health survey*. PLOS ONE.
- INEI (2024). *Encuesta Demográfica y de Salud Familiar — ENDES 2024*. Lima: Instituto Nacional de Estadística e Informática.
- OMS (2023). *Anaemia in women and children*. World Health Organization.

---

## 👤 Autor

**Kenny Vladim**  
📧 Contacto vía [GitHub](https://github.com/kennyvladim)

---

<p align="center">
  Desarrollado con ❤️ para la salud pública infantil en Perú
</p>
