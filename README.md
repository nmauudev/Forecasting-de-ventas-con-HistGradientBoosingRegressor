# 📊 Simulador de Ventas - Noviembre 2025

Aplicación de Machine Learning para simular y predecir las ventas de noviembre 2025 por producto, con análisis de escenarios de descuento y competencia.

**Modelo**: `HistGradientBoostingRegressor` con predicción recursiva día a día.

---

## 🏗️ Estructura del Proyecto

```
Forecasting de ventas/
├── .github/
│   └── workflows/
│       └── tests.yml            # CI: Ejecuta pytest en cada push
├── app/
│   ├── app.py                   # Aplicación Streamlit
│   └── requirements.txt         # Dependencias de la app
├── data/
│   ├── raw/                     # Datos originales (ignorados por git)
│   └── processed/
│       └── inferencia_df_transformado.csv  # Datos de inferencia
├── models/
│   └── modelo_final.joblib      # Modelo ML entrenado
├── notebooks/                   # Jupyter notebooks de exploración
├── tests/
│   └── test_feature_store.py    # Tests pytest del Feature Store
├── .dockerignore                # Archivos excluidos del build
├── .gitignore
├── docker-compose.yml           # Orquestación de contenedores
├── Dockerfile                   # Imagen Docker de la app
├── pytest.ini                   # Configuración de pytest
└── README.md
```

---

## 🐳 Despliegue con Docker

### Opción 1: Docker Compose (Recomendado)

```bash
# Levantar la aplicación (primera vez, construye la imagen)
docker-compose up --build

# Levantar en modo background (producción)
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f streamlit-app

# Detener
docker-compose down
```

### Opción 2: Docker directamente

```bash
# Construir la imagen
docker build -t simulador-ventas:latest .

# Ejecutar el contenedor
docker run -p 8501:8501 simulador-ventas:latest
```

La aplicación estará disponible en: **http://localhost:8501**

---

## 🧪 Tests - Feature Store

Los tests garantizan que la lógica del Feature Store (predicciones recursivas, actualización de lags, cálculo de precios) no se rompa al subir código nuevo.

### Ejecutar tests localmente

```bash
# Instalar dependencias de testing
pip install pytest pytest-cov

# Correr todos los tests
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ -v --cov=app --cov-report=term-missing

# Generar reporte HTML
pytest tests/ -v --html=reports/test-report.html
```

### Tests incluidos

| Test | Descripción |
|------|-------------|
| `test_output_tiene_columnas_predichas` | Verifica que el output tenga las columnas requeridas |
| `test_cantidad_filas_preservada` | El resultado tiene el mismo nº de días que la entrada |
| `test_predicciones_no_negativas` | Las unidades predichas nunca son negativas |
| `test_ingresos_calculados_correctamente` | `ingresos = unidades × precio_venta` |
| `test_ajuste_descuento_modifica_precio_venta` | Un mayor descuento reduce el precio |
| `test_escenario_competencia_modifica_precios_rivales` | El escenario ajusta Amazon, Decathlon, Deporvillage |
| `test_descuento_limitado_entre_menos100_y_100` | El descuento está limitado a [-100, 100] |
| `test_ratio_precio_recalculado` | `ratio = precio_venta / precio_competencia` |
| `test_modelo_predict_llamado_n_veces` | `predict()` se llama exactamente 1 vez por día |
| `test_dataframe_original_no_modificado` | La función trabaja con copias, no modifica el original |
| Tests de validaciones (`TestFeatureStoreValidaciones`) | Fórmulas matemáticas y estructura del CSV |

---

## ⚙️ CI/CD - GitHub Actions

El workflow `.github/workflows/tests.yml` se ejecuta automáticamente en cada **push** y **pull request** a `main`/`master`.

```
push a main → GitHub Actions:
  1. Checkout del código
  2. Configura Python 3.10 y 3.11
  3. Instala dependencias
  4. Ejecuta pytest con cobertura
  5. Sube reporte de cobertura como artifact
```

---

## 🚀 Instalación Local (sin Docker)

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# 2. Instalar dependencias
pip install -r app/requirements.txt

# 3. Ejecutar la aplicación
streamlit run app/app.py
```

---

## 📊 Funcionalidades del Dashboard

- **Selector de producto**: 24 productos disponibles para noviembre 2025
- **Ajuste de descuento**: Slider -50% a +50% con impacto en tiempo real
- **Escenaros de competencia**: Simula subidas/bajadas de precios en Amazon, Decathlon y Deporvillage
- **Predicción recursiva**: Actualiza lags día a día para una simulación realista
- **Comparativa de escenarios**: Compara los 3 escenarios de competencia en paralelo
- **Black Friday (28/11)**: Destacado especialmente en gráficos y tablas
