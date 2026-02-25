# 📊 Simulador de Ventas - Noviembre 2025

Aplicación de Streamlit para simular y visualizar predicciones de ventas de noviembre 2025 utilizando un modelo de Machine Learning (HistGradientBoostingRegressor).

## ✨ Características Principales

### 🎛️ Controles de Simulación (Sidebar)
- **Selector de Producto**: Dropdown con los 24 productos disponibles
- **Ajuste de Descuento**: Slider de -50% a +50% (pasos de 5%)
- **Escenario de Competencia**: 3 opciones (Actual 0%, Competencia -5%, Competencia +5%)
- **Botón de Simulación**: Ejecuta las predicciones recursivas

### 📈 Dashboard Principal

#### 1. KPIs Destacados
Cuatro tarjetas con métricas clave:
- 🛒 **Unidades Totales Proyectadas**
- 💵 **Ingresos Proyectados**
- 💰 **Precio Promedio de Venta**
- 🏷️ **Descuento Promedio**

#### 2. Gráfico de Predicción Diaria
- Línea única mostrando unidades vendidas predichas (días 1-30)
- **Black Friday (día 28)** marcado con:
  - Línea vertical destacada
  - Punto resaltado en rojo
  - Anotación clara con cantidad de unidades
- Diseñado con Seaborn para visualización profesional

#### 3. Tabla Detallada
Tabla interactiva con todos los días de noviembre:
- Fecha y día de la semana
- Precio de venta
- Precio de competencia
- Descuento aplicado
- Unidades predichas
- Ingresos proyectados por día
- **Fila del Black Friday destacada visualmente**

#### 4. Comparativa de Escenarios
Comparación de 3 escenarios de competencia:
- Escenario Base (0%)
- Competencia -5%
- Competencia +5%
- Muestra unidades totales e ingresos con deltas

## 🔄 Lógica de Predicciones Recursivas

La aplicación implementa predicciones día por día con actualización de lags:

1. **Día 1**: Usa los lags iniciales del archivo (calculados desde octubre)
2. **Día 2-30**: Para cada día:
   - Actualiza `unidades_vendidas_lag_1` con la predicción del día anterior
   - Desplaza lags: `lag_2 ← lag_1_anterior`, `lag_3 ← lag_2_anterior`, etc.
   - Actualiza `unidades_vendidas_ma7` con promedio de últimas 7 predicciones
   - Predice el día con lags actualizados

### Variables Recalculadas Dinámicamente
- `precio_venta`: Según descuento ajustado sobre `precio_base`
- `precio_competencia`: Ajusta Amazon, Decathlon y Deporvillage según escenario
- `descuento_porcentaje`: Recalculado con el ajuste del usuario
- `ratio_precio`: `precio_venta / precio_competencia`

## 🎨 Diseño Visual

- **Paleta de colores**: Gradientes morado/azul (#667eea, #764ba2)
- **Sidebar**: Fondo con gradiente morado
- **Tarjetas de métricas**: Diseño moderno con colores destacados
- **Gráficos**: Limpios y profesionales con Seaborn
- **Tablas**: Formato responsive con destacado de Black Friday
- **Animaciones**: Efectos hover en botones

## 🚀 Cómo Ejecutar

### 1. Instalar Dependencias
```bash
pip install -r app/requirements.txt
```

### 2. Ejecutar la Aplicación
```bash
streamlit run app/app.py
```

### 3. Abrir en el Navegador
La aplicación se abrirá automáticamente en:
- **Local**: http://localhost:8501
- **Network**: http://192.168.0.103:8501

## 📁 Estructura de Archivos

```
DS proyecto/
├── app/
│   ├── app.py              # Aplicación principal de Streamlit
│   └── requirements.txt    # Dependencias del proyecto
├── models/
│   └── modelo_final.joblib # Modelo ML entrenado
└── data/
    └── processed/
        └── inferencia_df_transformado.csv  # Datos de inferencia
```

## 📊 Datos de Entrada

El archivo `inferencia_df_transformado.csv` contiene:
- **24 productos** para noviembre 2025
- **30 días** de datos (1-30 de noviembre)
- **Todas las transformaciones ya aplicadas**:
  - Variables temporales (año, mes, día, semana, etc.)
  - Variables de lag (lag_1 a lag_7) inicializadas desde octubre
  - Media móvil (ma7)
  - Variables de precio (descuento, competencia, ratio)
  - One-hot encoding (nombre, categoría, subcategoría)

## 🔧 Tecnologías Utilizadas

- **Streamlit** 1.28.0+: Framework de la aplicación
- **Pandas** 2.0.0+: Manipulación de datos
- **NumPy** 1.24.0+: Operaciones numéricas
- **Joblib** 1.3.0+: Carga del modelo
- **Scikit-learn** 1.3.0+: Modelo de ML
- **Seaborn** 0.12.0+: Visualizaciones
- **Matplotlib** 3.7.0+: Gráficos base

## ⚠️ Notas Importantes

1. **Lags Iniciales**: El día 1 de noviembre ya tiene sus lags correctos (calculados desde octubre)
2. **Predicciones Recursivas**: Los lags se actualizan día a día con las predicciones previas
3. **No hay datos de octubre**: El archivo solo contiene noviembre, pero los lags fueron pre-calculados
4. **Black Friday**: El 28 de noviembre tiene descuento especial del 15% en el archivo original
5. **Validación**: El modelo espera exactamente las columnas en `modelo.feature_names_in_`

## 💡 Consejos de Uso

1. **Experimenta con descuentos**: Prueba diferentes ajustes para ver el impacto en ventas
2. **Compara escenarios**: Usa la sección de comparativa para decisiones estratégicas
3. **Observa el Black Friday**: Nota cómo el descuento especial impacta las ventas
4. **Analiza la tabla**: Revisa día por día para identificar patrones

## 🎯 Funcionalidades Implementadas

✅ Sidebar con todos los controles solicitados
✅ 4 KPIs destacados en tarjetas
✅ Gráfico de línea única con Black Friday marcado
✅ Tabla detallada con formato y destacado
✅ Comparativa de 3 escenarios de competencia
✅ Predicciones recursivas día por día
✅ Actualización de lags y media móvil
✅ Recálculo de variables dependientes
✅ Diseño moderno con gradientes morado/azul
✅ Manejo de errores y validaciones
✅ Spinners durante procesamiento
✅ Formato de números (2 decimales precios, 0 unidades)
✅ Mensajes informativos para el usuario

## 📝 Autor

Aplicación desarrollada para simulación de ventas con Machine Learning.
Modelo: HistGradientBoostingRegressor

---

**¡Disfruta simulando tus ventas de noviembre 2025!** 🚀
