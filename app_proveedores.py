import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import re

# ========================= CONFIGURACIÓN =========================
st.set_page_config(
    page_title="Evaluación de Facturas - Compatible con Modelo",
    page_icon="📋",
    layout="wide"
)

# VALORES DE REFERENCIA DE TU ENTRENAMIENTO (del análisis que hiciste)
QUANTILE_80_MONTO = 4907007  # Percentil 80 de tus datos reales


# ========================= AUTENTICACIÓN =========================
def check_login():
    """Sistema de autenticación simple"""
    users = {
        "proveedor1": "pass123",
        "mauricio": "mauricio123",
        "esteban": "esteban123",
        "Finz": "Finz123",
        "proveedor2": "pass456",
        "cliente_demo": "demo2024",
        "factoring_xyz": "fact789",
        "admin": "admin2024"
    }

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Sistema de Pre-evaluación de Facturas")
        st.markdown("---")

        with st.form("login_form"):
            st.subheader("Inicia Sesión")
            username = st.text_input("Usuario:")
            password = st.text_input("Contraseña:", type="password")
            login_button = st.form_submit_button("Ingresar")

            if login_button:
                if username in users and users[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Acceso autorizado")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")

        with st.expander("Información de Acceso"):
            st.markdown("**Usuario de prueba:** cliente_demo / demo2024")
        return False

    return True


def logout():
    """Función para cerrar sesión"""
    st.session_state.authenticated = False
    if "username" in st.session_state:
        del st.session_state.username
    st.rerun()


# ========================= CARGA DEL MODELO =========================
@st.cache_resource
def load_model():
    """Carga el modelo entrenado usando joblib"""
    try:
        model_paths = ['modelo_solo.pkl']
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                return model
        return None
    except Exception as e:
        return None


# ========================= PROCESAMIENTO COMPATIBLE CON TU MODELO =========================
def prepare_features_compatible_with_model(df):
    """
    Prepara las features EXACTAMENTE igual que tu modelo de entrenamiento
    Convierte de las columnas del Excel a las 5 features que usa tu modelo
    """
    try:
        # Crear DataFrame con las 5 features exactas de tu modelo
        features_df = pd.DataFrame()

        # 1. monto_numerico - directamente del monto de factura
        if 'monto_custodia' in df.columns:
            features_df['monto_numerico'] = pd.to_numeric(df['monto_custodia'], errors='coerce').fillna(0)
        else:
            features_df['monto_numerico'] = 0

        # 2. monto_alto - usando el mismo percentil 80 de tu entrenamiento
        features_df['monto_alto'] = (features_df['monto_numerico'] > QUANTILE_80_MONTO).astype(int)

        # 3. dias_emision_alto - basado en dias_desde_emision
        if 'dias_desde_emision' in df.columns:
            dias_emision = pd.to_numeric(df['dias_desde_emision'], errors='coerce').fillna(5)
        else:
            dias_emision = 5
        features_df['dias_emision_alto'] = (dias_emision > 5).astype(int)

        # 4. dias_venc_corto - basado en dias_hasta_vencimiento
        if 'dias_hasta_vencimiento' in df.columns:
            dias_venc = pd.to_numeric(df['dias_hasta_vencimiento'], errors='coerce').fillna(30)
        else:
            dias_venc = 30
        features_df['dias_venc_corto'] = (dias_venc < 30).astype(int)

        # 5. tiene_merito - basado en merito_ejecutivo
        if 'merito_ejecutivo' in df.columns:
            merito = pd.to_numeric(df['merito_ejecutivo'], errors='coerce').fillna(0)
        else:
            merito = 0
        features_df['tiene_merito'] = (merito > 0).astype(int)

        return features_df

    except Exception as e:
        st.error(f"Error preparando features: {str(e)}")
        return None


# ========================= FUNCIONES DE CONVERSIÓN (mantenidas de tu código) =========================
def convert_to_number(series):
    """Convierte cualquier cosa a número"""

    def to_num(value):
        if pd.isna(value) or value == '':
            return np.nan
        try:
            if isinstance(value, str):
                clean = value.replace('$', '').replace(',', '').replace(' ', '')
                return float(clean)
            else:
                return float(value)
        except:
            return np.nan

    return series.apply(to_num)


def convert_to_date(series):
    """Convierte cualquier cosa a fecha"""

    def to_date(value):
        if pd.isna(value) or value == '':
            return np.nan
        try:
            if isinstance(value, (int, float)) and 1 <= value <= 100000:
                base_date = datetime(1899, 12, 30)
                return base_date + timedelta(days=value)
            else:
                return pd.to_datetime(value, errors='coerce', dayfirst=True)
        except:
            return np.nan

    return series.apply(to_date)


def convert_columns_by_name(df):
    """Encuentra columnas por nombre y las convierte"""
    if df is None or df.empty:
        return df

    df_converted = df.copy()

    for col in df_converted.columns:
        try:
            col_name = str(col).upper().strip()

            if 'FOLIO' in col_name:
                df_converted[col] = convert_to_number(df_converted[col])
            elif any(word in col_name for word in ['MONTO', 'VALOR', 'PRECIO']):
                df_converted[col] = convert_to_number(df_converted[col])
            elif any(word in col_name for word in ['FECHA', 'DATE']):
                df_converted[col] = convert_to_date(df_converted[col])
        except Exception as e:
            continue

    return df_converted


# ========================= PROCESAMIENTO DE EXCEL =========================
def process_excel_simple(uploaded_file):
    """Lee Excel y maneja el formato específico con headers en fila 2"""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)

        if len(df) > 1:
            df.columns = df.iloc[1]
            df = df.drop([0, 1]).reset_index(drop=True)

        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(how='all').reset_index(drop=True)
        df_converted = convert_columns_by_name(df)
        return df_converted, None

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        return None, error_msg


def map_columns(df):
    """Mapea las columnas del Excel a las variables necesarias"""
    column_mapping = {}
    column_names = [str(col).strip() for col in df.columns]

    if 'FOLIO' in column_names:
        column_mapping['Folio'] = 'FOLIO'

    rut_columns = [col for col in df.columns if str(col).strip() == 'RUT']
    if len(rut_columns) >= 2:
        column_mapping['rut_pagador'] = rut_columns[1]
    elif len(rut_columns) == 1:
        column_mapping['rut_pagador'] = rut_columns[0]

    if 'NOMBRE PAGADOR' in column_names:
        column_mapping['razon_social_pagador'] = 'NOMBRE PAGADOR'

    if 'MONTO AL VENCIMIENTO' in column_names:
        column_mapping['monto_custodia'] = 'MONTO AL VENCIMIENTO'

    if 'FECHA EMISION' in column_names:
        column_mapping['fecha_emision'] = 'FECHA EMISION'

    if 'FECHA VENCIMIENTO' in column_names:
        column_mapping['fecha_vencimiento'] = 'FECHA VENCIMIENTO'

    if 'MERITO EJECUTIVO' in column_names:
        column_mapping['merito_ejecutivo'] = 'MERITO EJECUTIVO'

    mapped_df = pd.DataFrame()
    for target_col, source_col in column_mapping.items():
        if source_col in df.columns:
            try:
                if isinstance(source_col, str):
                    mapped_df[target_col] = df[source_col]
            except Exception as e:
                continue

    return mapped_df, column_mapping


def calculate_days_difference(fecha_emision, fecha_vencimiento, reference_date=None):
    """Calcula días desde emisión y hasta vencimiento"""
    if reference_date is None:
        reference_date = datetime.now()

    dias_desde_emision = []
    dias_hasta_vencimiento = []

    for i in range(len(fecha_emision)):
        try:
            emision = pd.to_datetime(fecha_emision.iloc[i])
            vencimiento = pd.to_datetime(fecha_vencimiento.iloc[i])

            if pd.notna(emision):
                dias_desde = (reference_date - emision).days
                dias_desde_emision.append(max(0, dias_desde))
            else:
                dias_desde_emision.append(5)

            if pd.notna(vencimiento):
                dias_hasta = (vencimiento - reference_date).days
                dias_hasta_vencimiento.append(dias_hasta)
            else:
                dias_hasta_vencimiento.append(30)

        except:
            dias_desde_emision.append(5)
            dias_hasta_vencimiento.append(30)

    return dias_desde_emision, dias_hasta_vencimiento


# ========================= APLICACIÓN PRINCIPAL =========================
def main():
    # Verificar autenticación
    if not check_login():
        return

    # Header con información del usuario
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Pre-evaluación de Facturas")
    with col2:
        st.write(f"**Usuario:** {st.session_state.username}")
        if st.button("Cerrar Sesión", type="secondary"):
            logout()

    st.markdown("---")

    # Cargar modelo
    model = load_model()

    # Información para proveedores
    with st.expander("¿Cómo funciona?"):
        st.markdown("""
        **Sistema de pre-evaluación de facturas.**

        **Resultados:**
        - **A (Aprobada)**: Alta probabilidad de aprobación
        - **R (Rechazada)**: Baja probabilidad de aprobación
        """)

    # Upload de archivo
    st.subheader("Subir Archivo Excel")
    uploaded_file = st.file_uploader(
        "Selecciona el archivo Excel con las facturas:",
        type=['xlsx', 'xls'],
    )

    if uploaded_file:
        with st.spinner("Procesando archivo..."):
            # Procesar Excel
            df, error = process_excel_simple(uploaded_file)

            if error or df is None or df.empty:
                st.error("Error al procesar el archivo o archivo vacío.")
                return

            # Mapear columnas
            mapped_df, column_mapping = map_columns(df)


            # Completar datos faltantes
            if 'merito_ejecutivo' not in mapped_df.columns:
                mapped_df['merito_ejecutivo'] = 0

            # Calcular días
            if 'fecha_emision' in mapped_df.columns and 'fecha_vencimiento' in mapped_df.columns:
                dias_emision, dias_vencimiento = calculate_days_difference(
                    mapped_df['fecha_emision'],
                    mapped_df['fecha_vencimiento']
                )
                mapped_df['dias_desde_emision'] = dias_emision
                mapped_df['dias_hasta_vencimiento'] = dias_vencimiento
            else:
                mapped_df['dias_desde_emision'] = 5
                mapped_df['dias_hasta_vencimiento'] = 30

            # Limpiar datos
            mapped_df['Folio'] = pd.to_numeric(mapped_df.get('Folio', 0), errors='coerce').fillna(0)
            mapped_df['monto_custodia'] = pd.to_numeric(mapped_df.get('monto_custodia', 0), errors='coerce').fillna(0)

            # Convertir mérito ejecutivo SI/NO a 1/0
            if 'merito_ejecutivo' in mapped_df.columns:
                mapped_df['merito_ejecutivo'] = mapped_df['merito_ejecutivo'].apply(
                    lambda x: 1 if str(x).upper().strip() == 'SI' else 0
                )

            # Filtrar filas válidas
            valid_rows = mapped_df['monto_custodia'] > 0
            processed_df = mapped_df[valid_rows].copy().reset_index(drop=True)

            if processed_df.empty:
                st.error("No hay facturas válidas para procesar.")
                return

            st.success(f"{len(processed_df)} facturas válidas procesadas")

            # ===================== USAR EL MODELO REAL =====================
            st.subheader("Evaluando facturas...")

            if model is not None:
                # Preparar features compatibles con el modelo
                X = prepare_features_compatible_with_model(processed_df)

                if X is not None:
                    #st.write("Features preparadas:")
                    #st.dataframe(X.head())

                    # Hacer predicciones
                    try:
                        predicciones = model.predict(X)
                        probabilidades = model.predict_proba(X)

                        # Convertir predicciones a A/R
                        results = ['A' if p == 1 else 'R' for p in predicciones]
                        probs_A = [prob[1] for prob in probabilidades]

                        processed_df['Resultado'] = results
                        processed_df['Probabilidad_A'] = probs_A


                    except Exception as e:
                        st.error(f"Error en predicción: {str(e)}")
                        return
                else:
                    st.error("Error preparando features")
                    return
            else:
                st.error("No se pudo cargar el modelo")
                return

            # Mostrar resultados
            st.subheader("Resultados de Pre-evaluación")

            # Métricas resumen
            total = len(processed_df)
            aprobadas = sum(1 for r in results if r == 'A')
            rechazadas = sum(1 for r in results if r == 'R')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Facturas", total)
            with col2:
                st.metric("Aprobadas", aprobadas, f"{aprobadas / total * 100:.1f}%")
            with col3:
                st.metric("Rechazadas", rechazadas, f"{rechazadas / total * 100:.1f}%")

            # Tabla de resultados
            display_columns = ['Folio', 'rut_pagador', 'razon_social_pagador', 'Resultado']
            available_columns = [col for col in display_columns if col in processed_df.columns]

            if 'monto_custodia' in processed_df.columns:
                processed_df['Monto (MM)'] = (processed_df['monto_custodia'] / 1_000_000).round(2)

            result_df = processed_df[
                available_columns + (['Monto (MM)'] if 'Monto (MM)' in processed_df.columns else [])].copy()

            # Colorear resultados
            def highlight_results(val):
                if val == 'A':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'R':
                    return 'background-color: #f8d7da; color: #721c24'
                return ''

            styled_df = result_df.style.applymap(highlight_results, subset=['Resultado'])
            st.dataframe(styled_df, use_container_width=True, height=400)

            # Descargar resultados
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(result_df)
            st.download_button(
                label="Descargar Resultados",
                data=csv,
                file_name=f"evaluacion_facturas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            # Mostrar estadísticas del modelo
            with st.expander(" Estadísticas del Modelo"):
                if 'Probabilidad_A' in processed_df.columns:
                    st.write("**Distribución de Probabilidades:**")
                    st.write(f"- Probabilidad promedio de aprobación: {processed_df['Probabilidad_A'].mean():.3f}")
                    st.write(f"- Probabilidad máxima: {processed_df['Probabilidad_A'].max():.3f}")
                    st.write(f"- Probabilidad mínima: {processed_df['Probabilidad_A'].min():.3f}")


if __name__ == "__main__":
    main()