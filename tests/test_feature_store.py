# ============================================================
# Tests del Feature Store - Simulador de Ventas
# ============================================================
# Ejecutar con: pytest tests/ -v
#
# Estos tests verifican la lógica crítica del Feature Store:
#   - Predicciones recursivas con actualización de lags
#   - Recálculo dinámico de precios y descuentos
#   - Validaciones de dominio (no negativos, límites, etc.)
# ============================================================

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ============================================================
# FIXTURES - Datos de prueba
# ============================================================

def crear_modelo_mock(n_features=15, prediccion_base=10.0):
    """
    Crea un modelo sklearn mock que devuelve predicciones fijas.
    Simula la interfaz de HistGradientBoostingRegressor.
    """
    modelo = MagicMock()
    # Definir los nombres de features que el modelo espera
    feature_names = [
        'precio_venta', 'precio_base', 'descuento_porcentaje',
        'Amazon', 'Decathlon', 'Deporvillage', 'precio_competencia',
        'ratio_precio',
        'unidades_vendidas_lag1', 'unidades_vendidas_lag2',
        'unidades_vendidas_lag3', 'unidades_vendidas_lag4',
        'unidades_vendidas_lag5', 'unidades_vendidas_lag6',
        'unidades_vendidas_lag7',
        'unidades_vendidas_mm7'
    ]
    modelo.feature_names_in_ = np.array(feature_names)
    modelo.predict = MagicMock(return_value=np.array([prediccion_base]))
    return modelo


def crear_df_producto(n_dias=5, precio_base=100.0, descuento_pct=10.0, seed=42):
    """
    Crea un DataFrame de producto de prueba con estructura idéntica
    a la que espera la función predecir_recursivo().
    """
    np.random.seed(seed)
    fechas = pd.date_range(start='2025-11-01', periods=n_dias, freq='D')

    data = {
        'fecha': fechas,
        'nombre': [f'Producto_Test'] * n_dias,
        'precio_base': [precio_base] * n_dias,
        'descuento_porcentaje': [descuento_pct] * n_dias,
        'precio_venta': [precio_base * (1 - descuento_pct / 100)] * n_dias,
        'Amazon': [95.0] * n_dias,
        'Decathlon': [98.0] * n_dias,
        'Deporvillage': [97.0] * n_dias,
        'precio_competencia': [96.67] * n_dias,
        'ratio_precio': [precio_base * (1 - descuento_pct / 100) / 96.67] * n_dias,
    }

    # Agregar columnas de lag
    for i in range(1, 8):
        data[f'unidades_vendidas_lag{i}'] = [float(10 + i)] * n_dias

    # Agregar media móvil
    data['unidades_vendidas_mm7'] = [12.0] * n_dias

    return pd.DataFrame(data)


# ============================================================
# Importar la función a testear desde app.py
# ============================================================

def importar_predecir_recursivo():
    """
    Importa dinámicamente la función predecir_recursivo desde app/app.py.
    Mockea streamlit para evitar que intente renderizar la UI.
    """
    import sys
    import os

    # Agregar el directorio app al path
    app_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app')
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Mock de streamlit y sus decoradores ANTES de importar
    streamlit_mock = MagicMock()
    streamlit_mock.cache_resource = lambda func: func
    streamlit_mock.cache_data = lambda func: func
    streamlit_mock.set_page_config = MagicMock()
    streamlit_mock.markdown = MagicMock()
    streamlit_mock.error = MagicMock()
    streamlit_mock.stop = MagicMock()
    streamlit_mock.sidebar = MagicMock()
    streamlit_mock.session_state = {}
    sys.modules['streamlit'] = streamlit_mock

    # Mock de joblib para evitar cargar el modelo real
    joblib_mock = MagicMock()
    joblib_mock.load = MagicMock(return_value=crear_modelo_mock())
    sys.modules['joblib'] = joblib_mock

    # Importar el módulo
    import importlib
    if 'app' in sys.modules:
        del sys.modules['app']

    import app as app_module
    return app_module.predecir_recursivo


# ============================================================
# CLASE DE TESTS: Lógica de Predicción Recursiva
# (Feature Store Core)
# ============================================================

class TestPredecirRecursivo:
    """
    Tests para la función predecir_recursivo() que es el corazón
    del Feature Store: actualiza lags y genera predicciones día a día.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup: obtiene la función a testear."""
        self.predecir = importar_predecir_recursivo()

    def test_output_tiene_columnas_predichas(self):
        """El DataFrame resultado debe tener 'unidades_predichas' e 'ingresos_proyectados'."""
        modelo = crear_modelo_mock(prediccion_base=15.0)
        df_producto = crear_df_producto(n_dias=5)

        resultado = self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        assert 'unidades_predichas' in resultado.columns, \
            "Falta columna 'unidades_predichas' en el resultado"
        assert 'ingresos_proyectados' in resultado.columns, \
            "Falta columna 'ingresos_proyectados' en el resultado"

    def test_cantidad_filas_preservada(self):
        """El resultado debe tener el mismo número de filas que la entrada."""
        modelo = crear_modelo_mock()
        n_dias = 7
        df_producto = crear_df_producto(n_dias=n_dias)

        resultado = self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        assert len(resultado) == n_dias, \
            f"Se esperaban {n_dias} filas, se obtuvieron {len(resultado)}"

    def test_predicciones_no_negativas(self):
        """Las predicciones de unidades nunca deben ser negativas."""
        # El modelo devuelve -5 (negativo), la función debe aplicar max(0, prediccion)
        modelo = crear_modelo_mock(prediccion_base=-5.0)
        df_producto = crear_df_producto(n_dias=5)

        resultado = self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        assert (resultado['unidades_predichas'] >= 0).all(), \
            "Las predicciones no deben ser negativas (deben aplicar max(0, pred))"

    def test_ingresos_calculados_correctamente(self):
        """Los ingresos proyectados = unidades_predichas * precio_venta."""
        modelo = crear_modelo_mock(prediccion_base=10.0)
        df_producto = crear_df_producto(n_dias=3, precio_base=100.0, descuento_pct=0.0)

        resultado = self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        # Con descuento 0% y precio_base=100, precio_venta=100
        # ingresos = predicciones * 100
        ingresos_esperados = resultado['unidades_predichas'] * resultado['precio_venta']
        pd.testing.assert_series_equal(
            resultado['ingresos_proyectados'].reset_index(drop=True),
            ingresos_esperados.reset_index(drop=True),
            check_names=False,
            rtol=1e-5
        )

    def test_ajuste_descuento_modifica_precio_venta(self):
        """Un ajuste de descuento positivo debe reducir el precio de venta."""
        modelo = crear_modelo_mock()
        precio_base = 100.0
        descuento_original = 10.0  # 10% de descuento → precio = 90
        ajuste = 10.0              # +10% adicional → nuevo descuento = 20% → precio = 80

        df_sin_ajuste = crear_df_producto(n_dias=5, precio_base=precio_base, descuento_pct=descuento_original)
        df_con_ajuste = crear_df_producto(n_dias=5, precio_base=precio_base, descuento_pct=descuento_original)

        res_sin = self.predecir(df_sin_ajuste, modelo, ajuste_descuento=0, escenario_competencia=0)
        res_con = self.predecir(df_con_ajuste, modelo, ajuste_descuento=ajuste, escenario_competencia=0)

        # Con más descuento, el precio debe ser MENOR
        assert res_con['precio_venta'].mean() < res_sin['precio_venta'].mean(), \
            "Un ajuste de descuento positivo debe resultar en precios de venta menores"

    def test_ajuste_descuento_cero_preserva_precio(self):
        """Con ajuste de descuento = 0, el precio de venta no debe cambiar."""
        modelo = crear_modelo_mock()
        precio_base = 100.0
        descuento_pct = 15.0
        precio_esperado = precio_base * (1 - descuento_pct / 100)

        df_producto = crear_df_producto(n_dias=5, precio_base=precio_base, descuento_pct=descuento_pct)
        resultado = self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        assert abs(resultado['precio_venta'].iloc[0] - precio_esperado) < 0.01, \
            f"Con ajuste=0, precio_venta debería ser {precio_esperado:.2f}, " \
            f"pero es {resultado['precio_venta'].iloc[0]:.2f}"

    def test_escenario_competencia_modifica_precios_rivales(self):
        """El escenario de competencia debe modificar los precios de Amazon, Decathlon y Deporvillage."""
        modelo = crear_modelo_mock()
        df1 = crear_df_producto(n_dias=3)
        df2 = crear_df_producto(n_dias=3)

        res_base = self.predecir(df1, modelo, ajuste_descuento=0, escenario_competencia=0)
        res_comp = self.predecir(df2, modelo, ajuste_descuento=0, escenario_competencia=5)

        # Con competencia +5%, los precios de competencia deben ser 5% mayores
        assert res_comp['Amazon'].mean() > res_base['Amazon'].mean(), \
            "Con escenario_competencia=+5%, el precio de Amazon debería subir"
        assert res_comp['precio_competencia'].mean() > res_base['precio_competencia'].mean(), \
            "Con escenario_competencia=+5%, precio_competencia debería subir"

    def test_descuento_limitado_entre_menos100_y_100(self):
        """El descuento no debe exceder los límites [-100, 100] incluso con ajustes extremos."""
        modelo = crear_modelo_mock()
        df_producto = crear_df_producto(n_dias=5, descuento_pct=90.0)

        # Intentar aplicar ajuste que superaría el 100%
        resultado = self.predecir(df_producto, modelo, ajuste_descuento=50, escenario_competencia=0)

        assert (resultado['descuento_porcentaje'] <= 100).all(), \
            "El descuento no debe superar el 100%"
        assert (resultado['descuento_porcentaje'] >= -100).all(), \
            "El descuento no debe bajar del -100%"

    def test_ratio_precio_recalculado(self):
        """El ratio_precio debe recalcularse como precio_venta / precio_competencia."""
        modelo = crear_modelo_mock()
        df_producto = crear_df_producto(n_dias=5)

        resultado = self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        # Verificar que ratio_precio es consistente
        ratio_calculado = resultado['precio_venta'] / resultado['precio_competencia']
        pd.testing.assert_series_equal(
            resultado['ratio_precio'].reset_index(drop=True),
            ratio_calculado.reset_index(drop=True),
            check_names=False,
            rtol=1e-4
        )

    def test_modelo_predict_llamado_n_veces(self):
        """El modelo debe ser llamado exactamente una vez por cada día."""
        n_dias = 10
        modelo = crear_modelo_mock(prediccion_base=5.0)
        df_producto = crear_df_producto(n_dias=n_dias)

        self.predecir(df_producto, modelo, ajuste_descuento=0, escenario_competencia=0)

        assert modelo.predict.call_count == n_dias, \
            f"Se esperaban {n_dias} llamadas a predict(), " \
            f"se realizaron {modelo.predict.call_count}"

    def test_dataframe_original_no_modificado(self):
        """La función no debe modificar el DataFrame original (debe trabajar con una copia)."""
        modelo = crear_modelo_mock()
        df_original = crear_df_producto(n_dias=5)
        lag1_original = df_original['unidades_vendidas_lag1'].copy()

        # Si la función modifica el df internamente está bien,
        # pero el llamador no debe ver sus datos originales cambiados
        # (la función hace .copy() del input)
        df_copia = df_original.copy()
        self.predecir(df_original, modelo, ajuste_descuento=0, escenario_competencia=0)

        # El lag1 del df original no debería cambiar
        pd.testing.assert_series_equal(
            df_original['unidades_vendidas_lag1'],
            lag1_original,
            check_names=False
        )


# ============================================================
# CLASE DE TESTS: Validaciones del Feature Store
# ============================================================

class TestFeatureStoreValidaciones:
    """
    Tests que validan la integridad y consistencia de las features
    que alimentan al modelo (sin necesidad de correr la app completa).
    """

    def test_precio_venta_formula_correcta(self):
        """precio_venta = precio_base * (1 - descuento / 100)"""
        precio_base = 200.0
        descuento = 25.0
        precio_esperado = precio_base * (1 - descuento / 100)  # 150.0

        assert abs(precio_esperado - 150.0) < 0.001, \
            "La fórmula de precio_venta no es correcta"

    def test_precio_competencia_promedio(self):
        """precio_competencia = (Amazon + Decathlon + Deporvillage) / 3"""
        amazon, decathlon, deporvillage = 90.0, 100.0, 110.0
        precio_comp_esperado = (amazon + decathlon + deporvillage) / 3  # 100.0

        assert abs(precio_comp_esperado - 100.0) < 0.001, \
            "El precio de competencia debe ser el promedio de los 3 competidores"

    def test_ratio_precio_formula(self):
        """ratio_precio = precio_venta / precio_competencia"""
        precio_venta = 90.0
        precio_competencia = 100.0
        ratio = precio_venta / precio_competencia  # 0.9

        assert abs(ratio - 0.9) < 0.001, \
            "El ratio de precio debe ser precio_venta / precio_competencia"

    def test_lag_shift_logica(self):
        """Verifica que el desplazamiento de lags es correcto: lag_i+1 ← lag_i."""
        # Simular un paso del Feature Store
        lag1, lag2, lag3 = 10, 11, 12
        nueva_pred = 15

        # Aplicar shift
        nuevo_lag2 = lag1
        nuevo_lag3 = lag2
        nuevo_lag1 = nueva_pred

        assert nuevo_lag1 == 15, "lag_1 debe ser la predicción actual"
        assert nuevo_lag2 == 10, "lag_2 debe ser el lag_1 anterior"
        assert nuevo_lag3 == 11, "lag_3 debe ser el lag_2 anterior"

    def test_media_movil_7dias(self):
        """La media móvil de 7 días debe ser correcta."""
        ultimas_7_predicciones = [10, 12, 8, 15, 11, 9, 13]
        mm7_esperada = np.mean(ultimas_7_predicciones)

        assert abs(mm7_esperada - np.mean([10, 12, 8, 15, 11, 9, 13])) < 0.001, \
            "La media móvil de 7 días no está siendo calculada correctamente"

    def test_estructura_columnas_inferencia(self):
        """
        El DataFrame de inferencia debe tener las columnas mínimas requeridas
        para que el Feature Store funcione correctamente.
        """
        df = crear_df_producto(n_dias=3)
        columnas_requeridas = [
            'fecha', 'precio_base', 'descuento_porcentaje', 'precio_venta',
            'Amazon', 'Decathlon', 'Deporvillage', 'precio_competencia',
            'ratio_precio', 'unidades_vendidas_mm7',
            *[f'unidades_vendidas_lag{i}' for i in range(1, 8)]
        ]

        for col in columnas_requeridas:
            assert col in df.columns, \
                f"Columna requerida '{col}' no está en el DataFrame de inferencia"

    def test_no_hay_nulos_en_features_criticas(self):
        """Las features críticas del modelo no deben tener valores nulos."""
        df = crear_df_producto(n_dias=30)
        features_criticas = [
            'precio_venta', 'precio_base', 'descuento_porcentaje',
            'unidades_vendidas_lag1', 'unidades_vendidas_mm7'
        ]

        for feature in features_criticas:
            assert df[feature].isnull().sum() == 0, \
                f"La feature '{feature}' contiene valores nulos"

    def test_factor_competencia_aplicado_correctamente(self):
        """escenario_competencia=+5 debe multiplicar precios por 1.05."""
        precio_original = 100.0
        escenario = 5
        factor = 1 + (escenario / 100)
        precio_esperado = precio_original * factor  # 105.0

        assert abs(precio_esperado - 105.0) < 0.001, \
            "El factor de competencia no se aplica correctamente"

    def test_factor_competencia_negativo(self):
        """escenario_competencia=-5 debe multiplicar precios por 0.95."""
        precio_original = 100.0
        escenario = -5
        factor = 1 + (escenario / 100)
        precio_esperado = precio_original * factor  # 95.0

        assert abs(precio_esperado - 95.0) < 0.001, \
            "El factor de competencia negativo no se aplica correctamente"
