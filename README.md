# Machine Learning Project

## Estructura del Proyecto

```
├── data/
│   ├── raw/                 # Datos originales sin procesar
│   └── processed/           # Datos procesados y listos para modelar
├── notebooks/               # Jupyter notebooks para exploración y análisis
├── models/                  # Modelos entrenados y guardados
├── app/                     # Aplicación Streamlit
├── docs/                    # Documentación del proyecto
├── requirements.txt         # Dependencias de Python
├── .gitignore              # Archivos a ignorar en Git
└── README.md               # Este archivo
```

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación Streamlit:
```bash
streamlit run app/app.py
```

## Estructura de Carpetas

- **data/raw/**: Almacena los datos originales descargados
- **data/processed/**: Datos limpios y procesados listos para el modelado
- **notebooks/**: Jupyter notebooks para exploración, análisis y experimentos
- **models/**: Modelos entrenados guardados en formato .pkl, .joblib o .h5
- **app/**: Aplicación Streamlit para presentar resultados
- **docs/**: Documentación técnica y guías del proyecto
