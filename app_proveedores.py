import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import pickle
import os


# ========================= CARGA DEL MODELO =========================
@st.cache_resource
def load_model():
    """Carga el modelo entrenado de sklearn"""
    try:
        # Intentar cargar desde diferentes ubicaciones
        model_paths = ['modelo.pkl', 'model.pkl', './models/modelo.pkl']

        for path in model_paths:
            if os.path.exists(path):
                with open(path, 'rb') as file:
                    model = pickle.load(file)
                st.success(f"Modelo cargado desde: {path}")
                return model

        st.warning("No se encontró archivo del modelo. Usando lógica simplificada.")
        return None

    except Exception as e:
        st.error(f"Error cargando modelo: {str(e)}")
        return None


def prepare_features_for_model(df):
    """Prepara las features para el modelo de sklearn"""
    # Crear las features que tu modelo espera
    features = pd.DataFrame()

    # Features básicas que probablemente usa tu modelo
    features['monto_custodia'] = df['monto_custodia']
    features['dias_desde_emision'] = df['dias_desde_emision']
    features['dias_hasta_vencimiento'] = df['dias_hasta_vencimiento']
    features['merito_ejecutivo'] = df['merito_ejecutivo']
    features['tasa'] = df['tasa']
    features['folio'] = df['Folio']

    # Features derivadas que podrían estar en tu modelo
    features['monto_mm'] = features['monto_custodia'] / 1_000_000
    features['dias_total'] = features['dias_desde_emision'] + features['dias_hasta_vencimiento']
    features['tasa_anual'] = features['tasa'] * 12

    # Rellenar NaN con valores por defecto
    features = features.fillna(0)

    return features


def check_login():
    """Sistema de autenticación simple"""

    # Usuarios predefinidos (usuario: contraseña)
    users = {
        "proveedor1": "pass123",
        "proveedor2": "pass456",
        "cliente_demo": "demo2024",
        "factoring_xyz": "fact789",
        "admin": "admin2024"
    }

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Acceso al Sistema de Pre-evaluación")
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

        # Información para usuarios
        with st.expander("Información de Acceso"):
            st.markdown("""
            **Para obtener acceso:**
            - Contacta al administrador del sistema
            - Se te proporcionará un usuario y contraseña únicos
            - El acceso es solo para evaluación previa de facturas

            **Usuarios de prueba disponibles:**
            - cliente_demo / demo2024
            """)

        return False

    return True


def logout():
    """Función para cerrar sesión"""
    st.session_state.authenticated = False
    if "username" in st.session_state:
        del st.session_state.username
    st.rerun()


# ========================= FUNCIONES DE CONVERSIÓN =========================
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


def convert_rut(series):
    """Convierte cualquier cosa a RUT formateado"""

    def to_rut(value):
        if pd.isna(value) or value == '':
            return np.nan
        try:
            rut_str = str(value).strip().upper()
            if re.match(r'^\d{1,2}\.\d{3}\.\d{3}-[0-9K]$', rut_str):
                return rut_str
            clean_rut = re.sub(r'[.\s-]', '', rut_str)
            if 8 <= len(clean_rut) <= 9:
                number = clean_rut[:-1]
                dv = clean_rut[-1]
                if len(number) == 7:
                    return f"{number[:1]}.{number[1:4]}.{number[4:]}-{dv}"
                elif len(number) == 8:
                    return f"{number[:2]}.{number[2:5]}.{number[5:]}-{dv}"
            return str(value)
        except:
            return str(value)

    return series.apply(to_rut)


def convert_to_rate(series):
    """Convierte cualquier cosa a tasa decimal"""

    def to_rate(value):
        if pd.isna(value) or value == '':
            return np.nan
        try:
            if isinstance(value, str):
                clean = value.replace('%', '').replace(' ', '')
                rate_val = float(clean)
            else:
                rate_val = float(value)
            if rate_val > 1:
                return rate_val / 100
            else:
                return rate_val
        except:
            return np.nan

    return series.apply(to_rate)


# ========================= PROCESAMIENTO DE EXCEL =========================
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
            elif any(word in col_name for word in ['MONTO', 'VALOR', 'PRECIO', 'INTERES']):
                df_converted[col] = convert_to_number(df_converted[col])
            elif any(word in col_name for word in ['FECHA', 'DATE']):
                df_converted[col] = convert_to_date(df_converted[col])
            elif 'RUT' in col_name:
                df_converted[col] = convert_rut(df_converted[col])
            elif 'TASA' in col_name:
                df_converted[col] = convert_to_rate(df_converted[col])
        except Exception as e:
            continue

    return df_converted


def process_excel_simple(uploaded_file):
    """Función que lee Excel y maneja el formato específico con headers en fila 2"""
    try:
        # Leer Excel sin procesar headers automáticamente
        df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)

        # Para tu formato específico, los headers están en la fila 2 (índice 1)
        if len(df) > 1:
            # Usar la fila 2 como headers
            df.columns = df.iloc[1]
            # Eliminar las primeras 2 filas (títulos y headers)
            df = df.drop([0, 1]).reset_index(drop=True)

        # Limpiar nombres de columnas
        df.columns = df.columns.astype(str).str.strip()

        # Eliminar filas completamente vacías
        df = df.dropna(how='all').reset_index(drop=True)

        df_converted = convert_columns_by_name(df)
        return df_converted, None

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        return None, error_msg


# ========================= MAPEO DE COLUMNAS =========================
def map_columns(df):
    """Mapea las columnas del Excel a las variables necesarias para el modelo"""

    # Diccionario de mapeo flexible
    column_mapping = {}

    # Buscar columnas específicas por nombre exacto primero
    column_names = [str(col).strip() for col in df.columns]

    # Mapeo específico para tu formato
    if 'FOLIO' in column_names:
        column_mapping['Folio'] = 'FOLIO'

    # Para RUT del pagador, buscar la segunda columna RUT
    rut_columns = [col for col in df.columns if str(col).strip() == 'RUT']
    if len(rut_columns) >= 2:
        column_mapping['rut_pagador'] = rut_columns[1]  # Segunda columna RUT
    elif len(rut_columns) == 1:
        column_mapping['rut_pagador'] = rut_columns[0]  # Si solo hay una, usarla

    if 'NOMBRE PAGADOR' in column_names:
        column_mapping['razon_social_pagador'] = 'NOMBRE PAGADOR'

    if 'MONTO AL VENCIMIENTO' in column_names:
        column_mapping['monto_custodia'] = 'MONTO AL VENCIMIENTO'

    if 'FECHA EMISION' in column_names:
        column_mapping['fecha_emision'] = 'FECHA EMISION'

    if 'FECHA VENCIMIENTO' in column_names:
        column_mapping['fecha_vencimiento'] = 'FECHA VENCIMIENTO'

    if 'TASA' in column_names:
        column_mapping['tasa'] = 'TASA'

    if 'MERITO EJECUTIVO' in column_names:
        column_mapping['merito_ejecutivo'] = 'MERITO EJECUTIVO'

    # Crear DataFrame con columnas mapeadas
    mapped_df = pd.DataFrame()

    for target_col, source_col in column_mapping.items():
        if source_col in df.columns:
            try:
                # Verificar que source_col es una sola columna
                if isinstance(source_col, str):
                    mapped_df[target_col] = df[source_col]
            except Exception as e:
                continue

    return mapped_df, column_mapping


# ========================= FUNCIONES DEL MODELO =========================
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
                dias_desde_emision.append(5)  # Valor por defecto

            if pd.notna(vencimiento):
                dias_hasta = (vencimiento - reference_date).days
                dias_hasta_vencimiento.append(dias_hasta)
            else:
                dias_hasta_vencimiento.append(30)  # Valor por defecto

        except:
            dias_desde_emision.append(5)
            dias_hasta_vencimiento.append(30)

    return dias_desde_emision, dias_hasta_vencimiento


def VMF_condition_simple(data):
    """Versión simplificada de VMF para una sola factura"""
    VMF = ""

    # Condición 1: Días hasta vencimiento
    if not (data["dias_hasta_vencimiento"] >= 15 and data["dias_hasta_vencimiento"] < 120):
        VMF += "1"

    # Condición 2: Monto sin mérito
    if not ((data["monto_custodia"] < 50e6 and data["merito_ejecutivo"] == 0) or data["merito_ejecutivo"] == 1):
        VMF += "2"

    # Condición 3: Folio mayor a 50
    if not (data["Folio"] > 50):
        VMF += "3"

    return VMF


# ========================= ÁRBOL DE DECISIÓN CON MODELO =========================
class SimpleDecisionTree:
    """Versión que puede usar modelo sklearn o lógica simplificada"""

    def __init__(self, model=None):
        self.blacklist = []  # Por seguridad, vacía para proveedores
        self.whitelist = []  # Por seguridad, vacía para proveedores
        self.model = model

    def classify_invoice(self, row):
        """Clasifica una sola factura usando modelo o lógica simple"""

        # Si tenemos modelo, usarlo
        if self.model is not None:
            return self.classify_with_model(row)
        else:
            return self.classify_with_logic(row)

    def classify_with_model(self, row):
        """Clasificación usando modelo sklearn"""
        try:
            # Preparar features para el modelo
            features_df = prepare_features_for_model(pd.DataFrame([row]))

            # Hacer predicción
            prediction = self.model.predict(features_df)[0]

            # Si tu modelo devuelve probabilidades, usar predict_proba
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_df)[0]
                # Convertir probabilidad a A/R/P según tu lógica
                if prediction == 1:  # Assuming 1 = approved
                    return 'A' if proba[1] > 0.7 else 'P'
                else:
                    return 'R'

            # Si solo devuelve clase
            return 'A' if prediction == 1 else 'R'

        except Exception as e:
            st.error(f"Error en predicción del modelo: {str(e)}")
            return self.classify_with_logic(row)

    def classify_with_logic(self, row):
        """Clasificación con lógica simplificada (fallback)"""
        # Valores por defecto para variables que no tenemos
        row['VM'] = ""  # Sin historial
        row['Dicom'] = 0  # Sin información DICOM
        row['M_Cust'] = 0  # Sin cartera previa
        row['Mora_prom_Monto'] = 0  # Sin historial de mora
        row['Tramo'] = 12  # Tramo por defecto
        row['sector_pagador'] = "PRIVADO"  # Por defecto privado
        row['Grupo'] = "Null"  # Sin grupo asignado
        row['rut'] = row.get('rut_pagador', '12345678-9')  # Usar RUT del pagador

        # Calcular VMF
        VMF = VMF_condition_simple(row)
        row['VMF'] = VMF

        # Lógica de decisión simplificada
        result = self.simple_decision_logic(row, VMF)
        return result

    def simple_decision_logic(self, data, VMF):
        """Lógica de decisión simplificada"""

        # Condiciones de rechazo básicas

        # 1. Días desde emisión muy altos
        if data.get('dias_desde_emision', 0) > 10:
            return 'R'

        # 2. Facturas muy grandes sin mérito
        if data['monto_custodia'] >= 50e6 and data.get('merito_ejecutivo', 0) == 0:
            return 'R'

        # 3. VMF tiene condiciones problemáticas
        if '2' in VMF:  # Problema con monto sin mérito
            return 'R'

        # 4. Días hasta vencimiento muy cortos o muy largos
        if data['dias_hasta_vencimiento'] < 15 or data['dias_hasta_vencimiento'] > 120:
            return 'R'

        # 5. Folio muy pequeño (posible factura irregular)
        if data.get('Folio', 100) <= 50:
            return 'R'

        # Si pasa todos los filtros básicos
        if VMF == "" and data.get('VM', '') == "":
            return 'A'

        # Casos pendientes para análisis manual
        if VMF == "" and data['monto_custodia'] >= 25e6:
            return 'P'

        return 'R'  # Por defecto rechazar


# ========================= APLICACIÓN PRINCIPAL =========================
def main():
    st.set_page_config(
        page_title="Evaluación de Facturas - Vista Proveedor",
        page_icon="📋",
        layout="wide"
    )

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

    # Información para proveedores
    with st.expander("¿Cómo funciona?"):
        st.markdown("""
        **Esta herramienta permite evaluar si las facturas serían probablemente aprobadas o rechazadas antes de enviarlas.**

        **Resultados posibles:**
        -  **A (Aprobada)**: La factura cumple los criterios básicos
        -  **R (Rechazada)**: La factura no cumple algún criterio importante  
        -  **P (Pendiente)**: Requiere análisis manual adicional
        """)

    # Upload de archivo
    st.subheader("Subir Archivo Excel")
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo Excel con las facturas:",
        type=['xlsx', 'xls'],
    )

    if uploaded_file:
        with st.spinner("Procesando archivo..."):
            # Procesar Excel
            df, error = process_excel_simple(uploaded_file)

            if error:
                st.error(f"Error al procesar el archivo: {error}")
                return

            if df is None or df.empty:
                st.error("El archivo está vacío o no se pudo leer.")
                return

            # Mapear columnas
            mapped_df, column_mapping = map_columns(df)

            # Procesar datos para el modelo
            st.subheader("Procesando datos...")

            # Completar datos faltantes
            if 'merito_ejecutivo' not in mapped_df.columns:
                mapped_df['merito_ejecutivo'] = 0
            if 'tasa' not in mapped_df.columns:
                mapped_df['tasa'] = 0.02  # 2% por defecto

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
            mapped_df['Folio'] = pd.to_numeric(mapped_df['Folio'], errors='coerce').fillna(0)
            mapped_df['monto_custodia'] = pd.to_numeric(mapped_df['monto_custodia'], errors='coerce').fillna(0)

            # Convertir mérito ejecutivo SI/NO a 1/0
            if 'merito_ejecutivo' in mapped_df.columns:
                mapped_df['merito_ejecutivo'] = mapped_df['merito_ejecutivo'].apply(
                    lambda x: 1 if str(x).upper().strip() == 'SI' else 0
                )
            else:
                mapped_df['merito_ejecutivo'] = 0

            # Filtrar filas válidas
            valid_rows = mapped_df['monto_custodia'] > 0
            processed_df = mapped_df[valid_rows].copy().reset_index(drop=True)

            if processed_df.empty:
                st.error("No hay facturas válidas para procesar (montos > 0).")
                return

            st.success(f"{len(processed_df)} facturas válidas procesadas")

            # Cargar modelo de machine learning
            model = load_model()

            # Aplicar modelo
            st.subheader("Evaluando facturas...")

            tree = SimpleDecisionTree(model=model)
            results = []

            progress_bar = st.progress(0)
            for i, (_, row) in enumerate(processed_df.iterrows()):
                result = tree.classify_invoice(row)
                results.append(result)
                progress_bar.progress((i + 1) / len(processed_df))

            processed_df['Resultado'] = results

            # Mostrar resultados
            st.subheader("Resultados de Pre-evaluación")

            # Métricas resumen
            total = len(processed_df)
            aprobadas = sum(1 for r in results if r == 'A')
            rechazadas = sum(1 for r in results if r == 'R')
            pendientes = sum(1 for r in results if r == 'P')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Facturas", total)
            with col2:
                st.metric("Aprobadas", aprobadas, f"{aprobadas / total * 100:.1f}%")
            with col3:
                st.metric("Rechazadas", rechazadas, f"{rechazadas / total * 100:.1f}%")
            with col4:
                st.metric("Pendientes", pendientes, f"{pendientes / total * 100:.1f}%")

            # Tabla de resultados
            display_columns = ['Folio', 'rut_pagador', 'razon_social_pagador', 'monto_custodia', 'Resultado']
            available_columns = [col for col in display_columns if col in processed_df.columns]

            # Formatear monto para mostrar
            if 'monto_custodia' in processed_df.columns:
                processed_df['Monto (MM)'] = (processed_df['monto_custodia'] / 1_000_000).round(2)

            result_df = processed_df[available_columns + ['Monto (MM)']].copy()

            # Colorear resultados
            def highlight_results(val):
                if val == 'A':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'R':
                    return 'background-color: #f8d7da; color: #721c24'
                elif val == 'P':
                    return 'background-color: #fff3cd; color: #856404'
                return ''

            styled_df = result_df.style.applymap(highlight_results, subset=['Resultado'])
            st.dataframe(styled_df, use_container_width=True, height=400)

            # Descargar resultados
            @st.cache_data
            def convert_df_to_excel(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_excel(result_df)
            st.download_button(
                label="Descargar Resultados",
                data=csv,
                file_name=f"evaluacion_facturas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            if aprobadas > 0:
                st.success(f"{aprobadas} facturas tienen buena probabilidad de aprobación.")

            if pendientes > 0:
                st.info(f"{pendientes} facturas requieren análisis adicional.")


if __name__ == "__main__":
    main()