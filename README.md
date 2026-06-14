# Healthcare Medical Project

Proyecto de fin de master de IA y Big Data. Sistema completo de ingenieria de datos, machine learning y visualizacion para el ambito hospitalario sobre Azure Synapse Analytics.

**Master FP Dual - IA y Big Data, Curso 2025-2026**

---

## De que va

Dataset sintetico de Kaggle con 100.000 pacientes y 3.5M registros clinicos. Pipeline completo desde la ingesta hasta el modelo predictivo de readmision hospitalaria a 30 dias, pasando por control de calidad, EDA, feature engineering, Power BI y app web con chat IA.

---

## Estructura

```
├── documentacion/       # PDFs entregables del proyecto
│   ├── InformeIA.pdf    # Memoria del proyecto
│   ├── ModeloDatos.pdf  # Modelo dimensional y star schema
│   ├── PlanPruebas.pdf  # Plan de pruebas
│   ├── Presentation.pdf # Slides de la presentacion
│   └── DiagramaEstrella.png
│
├── notebooks/           # Notebooks del pipeline (orden ejecucion)
│   ├── 00_*             # Descarga desde Kaggle
│   ├── 01_* / 01b_*    # Ingesta Bronze + control calidad
│   ├── 02_*             # EDA
│   ├── 03_* / 03b_*    # Silver + recuperacion cuarentena
│   ├── 04_*             # Star Schema + Power BI
│   └── pipelines/       # JSON de ejecucion
│
├── ia/                  # Modelo predictivo
│   └── icu_predictive_model_.ipynb
│
├── powerbi/             # Dashboard Power BI
│   └── MedicalProyect Power BI.pbix
│
└── app/                 # App web Streamlit
    ├── app.py
    ├── Dockerfile
    └── requirements.txt
```

---

## Tecnologias

Azure Synapse Analytics, Data Lake Storage, Key Vault, Python, PySpark, Scikit-learn, XGBoost, Power BI, Streamlit, DuckDB, Docker, Hugging Face Spaces.
