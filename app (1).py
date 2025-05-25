import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("dataset_comp_ratio.csv")
df = df.dropna()
# --- Carga de datos (debes adaptar estas líneas a tu entorno) ---
df = pd.read_csv("dataset_comp_ratio.csv")
df2 = pd.read_csv("dataset_comp_ratio_test.csv")
df = df.dropna()
df2 = df2.dropna()
# Crear target categórico
def clasificar_ratio(r):
    if r < 1.3:
        return 'Bajo'
    elif r < 1.6:
        return 'Medio'
    elif r < 2.7:
        return 'Alto'
    else:
        return 'Genial'

df['CLASE_CUMPLIMIENTO'] = df['RATIO_CUMPLIMIENTO'].apply(clasificar_ratio)
df2['CLASE_CUMPLIMIENTO'] = df2['RATIO_CUMPLIMIENTO'].apply(clasificar_ratio)
# --- Definir columnas ---
feature_cols = [
    'NIVELSOCIOECONOMICO_DES',
    'ENTORNO_DES',
    'MTS2VENTAS_NUM',
    'PUERTASREFRIG_NUM',
    'SEGMENTO_MAESTRO_DESC',
    'LID_UBICACION_TIENDA',
    'dist_comp_directa',
    'num_comp_directa',
    'dist_comp_indirecta',
    'num_comp_indirecta',
]

# --- Preprocesador ---
columnas_cat = ['NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA']
columnas_num = list(set(feature_cols) - set(columnas_cat))
preprocesadores_por_entorno = {}
resultados_por_entorno = {}

hiperparametros_entorno = {
    'Hogar':   {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100},
    'Base':    {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100},
    'Receso':  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100},
}

# --- Entrenar modelos por entorno ---
for entorno in df['ENTORNO_DES'].unique():
    df_ent = df[df['ENTORNO_DES'] == entorno].copy()
    df2_ent = df2[df2['ENTORNO_DES'] == entorno].copy()
    if len(df_ent) < 30 or len(df2_ent) < 10:
        continue

    X_train = df_ent[feature_cols]
    y_train = df_ent['CLASE_CUMPLIMIENTO']

    preprocesador_local = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='mean'), columnas_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_cat)
    ])

    params = hiperparametros_entorno.get(entorno, hiperparametros_entorno['Base'])
    modelo = Pipeline([
        ('preprocesador', preprocesador_local),
        ('clasificador', RandomForestClassifier(random_state=42, class_weight='balanced', **params))
    ])

    modelo.fit(X_train, y_train)

    resultados_por_entorno[entorno] = {
        'modelo': modelo,
        'params': params,
        'preprocesador': preprocesador_local
    }

# Fallback para entornos con pocos datos
if 'Peatonal' not in resultados_por_entorno:
    resultados_por_entorno['Peatonal'] = resultados_por_entorno['Base']

# --- Mostrar hiperparámetros en Streamlit ---
st.sidebar.title("Hiperparámetros por entorno")
for entorno, datos in resultados_por_entorno.items():
    with st.sidebar.expander(f"{entorno}"):
        for k, v in datos['params'].items():
            st.write(f"{k}: {v}")

# --- Simulador de predicción ---
st.title("🎯 Simulador de predicción por entorno")
st.markdown("Ingresa los datos de una tienda y selecciona su entorno para predecir su clase de cumplimiento.")

input_data = {}

col1, col2 = st.columns(2)
with col1:
    entorno = st.selectbox("ENTORNO_DES", options=list(resultados_por_entorno.keys()))
    input_data['ENTORNO_DES'] = entorno
    input_data['NIVELSOCIOECONOMICO_DES'] = st.text_input("NIVELSOCIOECONOMICO_DES", "BC")
    input_data['SEGMENTO_MAESTRO_DESC'] = st.text_input("SEGMENTO_MAESTRO_DESC", "Parada Técnica")
    input_data['LID_UBICACION_TIENDA'] = st.text_input("LID_UBICACION_TIENDA", "UT_CARRETERA_GAS")

with col2:
    input_data['MTS2VENTAS_NUM'] = st.number_input("MTS2VENTAS_NUM", value=50.0)
    input_data['PUERTASREFRIG_NUM'] = st.number_input("PUERTASREFRIG_NUM", value=4)
    input_data['dist_comp_directa'] = st.number_input("dist_comp_directa", value=0.5)
    input_data['num_comp_directa'] = st.number_input("num_comp_directa", value=2)
    input_data['dist_comp_indirecta'] = st.number_input("dist_comp_indirecta", value=1.2)
    input_data['num_comp_indirecta'] = st.number_input("num_comp_indirecta", value=3)

if st.button("🔮 Predecir Clase de Cumplimiento"):
    try:
        modelo = resultados_por_entorno.get(entorno, resultados_por_entorno['Base'])['modelo']
        input_df = pd.DataFrame([input_data])[feature_cols]
        pred = modelo.predict(input_df)[0]
        st.success(f"La clase de cumplimiento estimada para este registro es: **{pred}**")
    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")

df3 = pd.concat([df, df2], ignore_index=True)

# --- Filtros ---
st.sidebar.header("🎛️ Filtros")
entornos = st.sidebar.multiselect("Entorno", df3["ENTORNO_DES"].unique(), default=df3["ENTORNO_DES"].unique())
segmentos = st.sidebar.multiselect("Segmento Maestro", df3["SEGMENTO_MAESTRO_DESC"].unique(), default=df3["SEGMENTO_MAESTRO_DESC"].unique())

df_filtrado = df3[df3["ENTORNO_DES"].isin(entornos) & df3["SEGMENTO_MAESTRO_DESC"].isin(segmentos)]

# --- KPIs ---
st.subheader("Indicadores clave")
col1, col2, col3 = st.columns(3)
col1.metric("Tiendas analizadas", len(df_filtrado))
col2.metric("Promedio de RATIO_CUMPLIMIENTO", round(df_filtrado["RATIO_CUMPLIMIENTO"].mean(), 2))
col3.metric("Tiendas debajo de la meta", (df_filtrado["RATIO_CUMPLIMIENTO"] < 1).sum())

# --- Mapa de desempeño ---
st.subheader("🗺️ Mapa de tiendas")
mapa = folium.Map(location=[df_filtrado["LATITUD_NUM"].mean(), df_filtrado["LONGITUD_NUM"].mean()], zoom_start=6)
cluster = MarkerCluster().add_to(mapa)

def color_ratio(r):
    if r < 0.9:
        return "red"
    elif r < 1.1:
        return "orange"
    elif r < 2:
        return "green"
    else:
        return "blue"

for _, row in df_filtrado.iterrows():
    popup = f"""
    <b>Tienda:</b> {row['TIENDA_ID']}<br>
    <b>Venta:</b> ${row['VENTA_TOTAL']:,.0f}<br>
    <b>Meta:</b> ${row['META_VENTA']:,.0f}<br>
    <b>Ratio:</b> {row['RATIO_CUMPLIMIENTO']:.2f}<br>
    <b>Entorno:</b> {row['ENTORNO_DES']}<br>
    <b>Segmento:</b> {row['SEGMENTO_MAESTRO_DESC']}<br>
    <b>Ubicación:</b> {row['LID_UBICACION_TIENDA']}
    """
    folium.CircleMarker(
        location=[row["LATITUD_NUM"], row["LONGITUD_NUM"]],
        radius=5,
        color=color_ratio(row["RATIO_CUMPLIMIENTO"]),
        fill=True,
        fill_color=color_ratio(row["RATIO_CUMPLIMIENTO"]),
        fill_opacity=0.7,
        popup=popup
    ).add_to(cluster)

st_data = st_folium(mapa, width=700, height=500)

# --- Tabla final ---
st.subheader("📄 Datos filtrados")
st.dataframe(df_filtrado.reset_index(drop=True))

# --- Definir columnas ---
feature_cols = [
    'NIVELSOCIOECONOMICO_DES',
    'ENTORNO_DES',
    'MTS2VENTAS_NUM',
    'PUERTASREFRIG_NUM',
    'SEGMENTO_MAESTRO_DESC',
    'LID_UBICACION_TIENDA',
    'dist_comp_directa',
    'num_comp_directa',
    'dist_comp_indirecta',
    'num_comp_indirecta',
]
