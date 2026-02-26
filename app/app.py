import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Simulador de Ventas - Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Paleta de colores morada/azul */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
    }
    
    /* Estilo del sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Títulos principales */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Tarjetas de métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tablas */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Separadores */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo = joblib.load('models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

@st.cache_data
def cargar_datos():
    """Carga el dataframe de inferencia"""
    try:
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        st.stop()

def predecir_recursivo(df_producto, modelo, ajuste_descuento, escenario_competencia):
    """
    Realiza predicciones recursivas día por día actualizando los lags.
    
    Args:
        df_producto: DataFrame filtrado para un producto específico
        modelo: Modelo entrenado
        ajuste_descuento: Porcentaje de ajuste al descuento (-50 a +50)
        escenario_competencia: Ajuste a precios de competencia (0, -5, +5)
    
    Returns:
        DataFrame con predicciones y variables actualizadas
    """

    df = df_producto.copy().sort_values('fecha').reset_index(drop=True)
    
    
    predicciones = []
    
    lag_cols = [f'unidades_vendidas_lag{i}' for i in range(1, 8)]
    ma_col = 'unidades_vendidas_mm7'
    
    for idx in range(len(df)):
        fila = df.iloc[idx].copy()
        
        
        precio_base = fila['precio_base']
        descuento_original = fila['descuento_porcentaje']
        
        
        nuevo_descuento = descuento_original + ajuste_descuento
        nuevo_descuento = max(-100, min(100, nuevo_descuento))  # Limitar entre -100 y 100
        
        nuevo_precio_venta = precio_base * (1 - nuevo_descuento / 100)
        
        
        fila['precio_venta'] = nuevo_precio_venta
        fila['descuento_porcentaje'] = nuevo_descuento
        
        
        factor_competencia = 1 + (escenario_competencia / 100)
        fila['Amazon'] = df.iloc[idx]['Amazon'] * factor_competencia
        fila['Decathlon'] = df.iloc[idx]['Decathlon'] * factor_competencia
        fila['Deporvillage'] = df.iloc[idx]['Deporvillage'] * factor_competencia
        
        
        fila['precio_competencia'] = (fila['Amazon'] + fila['Decathlon'] + fila['Deporvillage']) / 3
        
        
        if fila['precio_competencia'] > 0:
            fila['ratio_precio'] = fila['precio_venta'] / fila['precio_competencia']
        else:
            fila['ratio_precio'] = 1.0
        
        
        features = fila[modelo.feature_names_in_].values.reshape(1, -1)
        
        
        prediccion = modelo.predict(features)[0]
        prediccion = max(0, prediccion)  # No permitir predicciones negativas
        
        # Guardar predicción
        predicciones.append(prediccion)
        
       
        if idx < len(df) - 1:
            # Desplazar lags: lag_7 <- lag_6, lag_6 <- lag_5, ..., lag_1 <- predicción actual
            for i in range(6, 0, -1):
                df.at[idx + 1, f'unidades_vendidas_lag{i+1}'] = df.at[idx, f'unidades_vendidas_lag{i}']
            
            
            df.at[idx + 1, 'unidades_vendidas_lag1'] = prediccion
            
            
            if idx >= 6:
                ultimas_7 = predicciones[-7:]
            else:
                
                ultimas_7 = predicciones.copy()
                for i in range(1, 8 - len(predicciones)):
                    if f'unidades_vendidas_lag{i}' in df.columns:
                        ultimas_7.insert(0, df.at[idx, f'unidades_vendidas_lag{i}'])
                ultimas_7 = ultimas_7[:7]
            
            df.at[idx + 1, ma_col] = np.mean(ultimas_7)
        
        
        df.at[idx, 'precio_venta'] = nuevo_precio_venta
        df.at[idx, 'descuento_porcentaje'] = nuevo_descuento
        df.at[idx, 'Amazon'] = fila['Amazon']
        df.at[idx, 'Decathlon'] = fila['Decathlon']
        df.at[idx, 'Deporvillage'] = fila['Deporvillage']
        df.at[idx, 'precio_competencia'] = fila['precio_competencia']
        df.at[idx, 'ratio_precio'] = fila['ratio_precio']
    
    
    df['unidades_predichas'] = predicciones
    df['ingresos_proyectados'] = df['unidades_predichas'] * df['precio_venta']
    
    return df


modelo = cargar_modelo()
df_inferencia = cargar_datos()


st.sidebar.markdown("# 🎛️ Controles de Simulación")
st.sidebar.markdown("---")


productos = sorted(df_inferencia['nombre'].unique())


producto_seleccionado = st.sidebar.selectbox(
    "📦 Seleccionar Producto",
    productos,
    help="Elige el producto para simular sus ventas"
)


ajuste_descuento = st.sidebar.slider(
    "💰 Ajuste de Descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el descuento del producto (positivo = más descuento, negativo = menos descuento)"
)


st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario_competencia = st.sidebar.radio(
    "Ajuste de precios de la competencia:",
    options=[0, -5, 5],
    format_func=lambda x: f"Actual (0%)" if x == 0 else f"Competencia {x:+d}%",
    help="Simula cambios en los precios de Amazon, Decathlon y Deporvillage"
)

st.sidebar.markdown("---")


simular = st.sidebar.button("🚀 Simular Ventas", use_container_width=True)


st.markdown("# 📊 Dashboard de Simulación de Ventas")
st.markdown(f"### Predicciones para **{producto_seleccionado}** - Noviembre 2025")
st.markdown("---")


if simular or 'df_resultado' not in st.session_state:
    with st.spinner('🔄 Procesando predicciones recursivas...'):
        # Filtrar datos para el producto seleccionado
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        # Realizar predicción recursiva
        df_resultado = predecir_recursivo(df_producto, modelo, ajuste_descuento, escenario_competencia)
        
        # Guardar en session_state
        st.session_state['df_resultado'] = df_resultado
        st.session_state['producto'] = producto_seleccionado
        st.session_state['ajuste_descuento'] = ajuste_descuento
        st.session_state['escenario_competencia'] = escenario_competencia

# Obtener resultados
if 'df_resultado' in st.session_state:
    df_resultado = st.session_state['df_resultado']
    
    # 1. KPIs DESTACADOS
    st.markdown("## 📈 Métricas Clave")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unidades_totales = df_resultado['unidades_predichas'].sum()
        st.metric(
            label="🛒 Unidades Totales",
            value=f"{unidades_totales:,.0f}",
            help="Total de unidades proyectadas para noviembre"
        )
    
    with col2:
        ingresos_totales = df_resultado['ingresos_proyectados'].sum()
        st.metric(
            label="💵 Ingresos Proyectados",
            value=f"€{ingresos_totales:,.2f}",
            help="Ingresos totales estimados"
        )
    
    with col3:
        precio_promedio = df_resultado['precio_venta'].mean()
        st.metric(
            label="💰 Precio Promedio",
            value=f"€{precio_promedio:.2f}",
            help="Precio de venta promedio"
        )
    
    with col4:
        descuento_promedio = df_resultado['descuento_porcentaje'].mean()
        st.metric(
            label="🏷️ Descuento Promedio",
            value=f"{descuento_promedio:.1f}%",
            help="Descuento promedio aplicado"
        )
    
    st.markdown("---")
    
    # 2. GRÁFICO DE PREDICCIÓN DIARIA
    st.markdown("## 📉 Predicción Diaria de Ventas")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Configurar estilo de seaborn
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Extraer día del mes para el eje X
    df_resultado['dia'] = df_resultado['fecha'].dt.day
    
    # Crear gráfico de línea
    sns.lineplot(
        data=df_resultado,
        x='dia',
        y='unidades_predichas',
        marker='o',
        linewidth=2.5,
        markersize=6,
        color='#667eea',
        ax=ax
    )
    
    # Marcar Black Friday (día 28)
    black_friday_idx = df_resultado[df_resultado['dia'] == 28].index[0]
    black_friday_ventas = df_resultado.loc[black_friday_idx, 'unidades_predichas']
    
    # Línea vertical para Black Friday
    ax.axvline(x=28, color='#764ba2', linestyle='--', linewidth=2, alpha=0.7, label='Black Friday')
    
    # Punto destacado en Black Friday
    ax.plot(28, black_friday_ventas, 'ro', markersize=12, zorder=5)
    
    # Anotación para Black Friday
    ax.annotate(
        f'🛍️ Black Friday\n{black_friday_ventas:.0f} unidades',
        xy=(28, black_friday_ventas),
        xytext=(28, black_friday_ventas * 1.15),
        ha='center',
        fontsize=11,
        fontweight='bold',
        color='#764ba2',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#764ba2', linewidth=2)
    )
    
    # Configurar etiquetas y título
    ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
    ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', fontsize=14, fontweight='bold', pad=20)
    
    # Configurar eje X para mostrar todos los días
    ax.set_xticks(range(1, 31))
    ax.set_xlim(0.5, 30.5)
    
    # Grid más sutil
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mejorar layout
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 3. TABLA DETALLADA
    st.markdown("## 📋 Detalle Diario de Predicciones")
    
    # Preparar tabla para mostrar
    df_tabla = df_resultado[['fecha', 'precio_venta', 'precio_competencia', 'descuento_porcentaje', 
                              'unidades_predichas', 'ingresos_proyectados']].copy()
    
    # Agregar día de la semana
    df_tabla['dia_semana'] = df_resultado['fecha'].dt.day_name()
    
    # Formatear fecha
    df_tabla['fecha_fmt'] = df_resultado['fecha'].dt.strftime('%d/%m/%Y')
    
    # Reordenar columnas
    df_tabla = df_tabla[['fecha_fmt', 'dia_semana', 'precio_venta', 'precio_competencia', 
                          'descuento_porcentaje', 'unidades_predichas', 'ingresos_proyectados']]
    
    # Renombrar columnas
    df_tabla.columns = ['Fecha', 'Día', 'Precio Venta (€)', 'Precio Competencia (€)', 
                        'Descuento (%)', 'Unidades', 'Ingresos (€)']
    
    # Formatear números
    df_tabla['Precio Venta (€)'] = df_tabla['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
    df_tabla['Precio Competencia (€)'] = df_tabla['Precio Competencia (€)'].apply(lambda x: f"€{x:.2f}")
    df_tabla['Descuento (%)'] = df_tabla['Descuento (%)'].apply(lambda x: f"{x:.1f}%")
    df_tabla['Unidades'] = df_tabla['Unidades'].apply(lambda x: f"{x:.0f}")
    df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].apply(lambda x: f"€{x:.2f}")
    
    # Destacar Black Friday
    def highlight_black_friday(row):
        if '28/11/2025' in row['Fecha']:
            return ['background-color: #fff3cd; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    # Aplicar estilo y mostrar
    st.dataframe(
        df_tabla.style.apply(highlight_black_friday, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Nota sobre Black Friday
    st.info("🛍️ **Black Friday (28 de noviembre)** destacado en amarillo")
    
    st.markdown("---")
    
    # 4. COMPARATIVA DE ESCENARIOS
    st.markdown("## 🔄 Comparativa de Escenarios de Competencia")
    st.markdown("*Manteniendo el ajuste de descuento actual y variando solo los precios de competencia*")
    
    # Calcular los 3 escenarios
    with st.spinner('Calculando escenarios...'):
        escenarios = {}
        for escenario in [0, -5, 5]:
            df_producto_temp = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
            df_escenario = predecir_recursivo(df_producto_temp, modelo, ajuste_descuento, escenario)
            escenarios[escenario] = {
                'unidades': df_escenario['unidades_predichas'].sum(),
                'ingresos': df_escenario['ingresos_proyectados'].sum()
            }
    
    # Mostrar comparativa en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Escenario Base")
        st.markdown("*Precios actuales (0%)*")
        st.metric("Unidades Totales", f"{escenarios[0]['unidades']:,.0f}")
        st.metric("Ingresos Totales", f"€{escenarios[0]['ingresos']:,.2f}")
    
    with col2:
        st.markdown("### 📉 Competencia -5%")
        st.markdown("*Competencia reduce precios*")
        delta_unidades = escenarios[-5]['unidades'] - escenarios[0]['unidades']
        delta_ingresos = escenarios[-5]['ingresos'] - escenarios[0]['ingresos']
        st.metric("Unidades Totales", f"{escenarios[-5]['unidades']:,.0f}", 
                 delta=f"{delta_unidades:+.0f}")
        st.metric("Ingresos Totales", f"€{escenarios[-5]['ingresos']:,.2f}",
                 delta=f"€{delta_ingresos:+,.2f}")
    
    with col3:
        st.markdown("### 📈 Competencia +5%")
        st.markdown("*Competencia aumenta precios*")
        delta_unidades = escenarios[5]['unidades'] - escenarios[0]['unidades']
        delta_ingresos = escenarios[5]['ingresos'] - escenarios[0]['ingresos']
        st.metric("Unidades Totales", f"{escenarios[5]['unidades']:,.0f}",
                 delta=f"{delta_unidades:+.0f}")
        st.metric("Ingresos Totales", f"€{escenarios[5]['ingresos']:,.2f}",
                 delta=f"€{delta_ingresos:+,.2f}")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>💡 <strong>Tip:</strong> Ajusta los controles en el sidebar y presiona "Simular Ventas" para ver nuevos resultados</p>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>
            Dashboard de Simulación de Ventas | Noviembre 2025 | 
            Modelo: HistGradientBoostingRegressor
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Mensaje inicial
    st.info("👈 Selecciona un producto y ajusta los parámetros en el sidebar, luego presiona **'Simular Ventas'** para comenzar.")
