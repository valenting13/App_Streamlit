import streamlit as st
import pandas as pd
import numpy as np
import joblib
from itertools import product
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import plotly.express as px

# Definir colores
COLOR_1 = '#D18205'  # Naranja
COLOR_2 = '#20255D'  # Azul oscuro
COLOR_3 = '#115EE3'  # Azul brillante
COLOR_4 = '#24C2F1'  # Azul claro
# Definir funciones personalizadas

# 1. Función para crear características de fecha
def create_date_features_manually(df):
    df['nro_mes'] = df['Fecha Pedido'].dt.month
    df['dia_mes'] = df['Fecha Pedido'].dt.day
    df['dia_anio'] = df['Fecha Pedido'].dt.dayofyear
    df['semana_anio'] = df['Fecha Pedido'].dt.isocalendar().week.astype(int)
    df['dia_semana'] = df['Fecha Pedido'].dt.dayofweek + 1
    df['anio'] = df['Fecha Pedido'].dt.year
    df["fin_semana"] = df['Fecha Pedido'].dt.weekday // 4
    df["trimestre"] = df['Fecha Pedido'].dt.quarter
    df['inicio_mes'] = df['Fecha Pedido'].dt.is_month_start.astype(int)
    df['fin_mes'] = df['Fecha Pedido'].dt.is_month_end.astype(int)
    df['inicio_trim'] = df['Fecha Pedido'].dt.is_quarter_start.astype(int)
    df['fin_trim'] = df['Fecha Pedido'].dt.is_quarter_end.astype(int)
    df['inicio_anio'] = df['Fecha Pedido'].dt.is_year_start.astype(int)
    df['fin_anio'] = df['Fecha Pedido'].dt.is_year_end.astype(int)
    df["estacion"] = np.where(df['nro_mes'].isin([12, 1, 2, 3, 4]), 2, 1)
    return df

# 2. Función para generar ruido aleatorio
def random_noise(dataframe):
    return np.random.normal(size=(len(dataframe),))

# 3. Función para generar características lag
def lag_features_fn(dataframe):
    return lag_features(dataframe, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

def lag_features(dataframe, lags):
    if isinstance(dataframe, np.ndarray):
        dataframe = pd.DataFrame(dataframe, columns=["País", "Categoría Tarjeta", "Ventas"] + [f'feature_{i}' for i in range(3, dataframe.shape[1])])
    for lag in lags:
        dataframe[f'Ventas_lag_{lag}'] = dataframe.groupby(["País", "Categoría Tarjeta"])['Ventas'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# 4. Función para generar rolling mean features
def roll_mean_features_fn(dataframe):
    return roll_mean_features(dataframe, [365, 546, 730])

def roll_mean_features(dataframe, windows):
    if isinstance(dataframe, np.ndarray):
        dataframe = pd.DataFrame(dataframe, columns=["País", "Categoría Tarjeta", "Ventas"] + [f'feature_{i}' for i in range(3, dataframe.shape[1])])
    for window in windows:
        dataframe[f'Ventas_roll_mean_{window}'] = dataframe.groupby(["País", "Categoría Tarjeta"])['Ventas'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe

# 5. Función para generar características EWM
def ewm_features_fn(dataframe):
    return ewm_features(dataframe, [0.99, 0.95, 0.9, 0.8, 0.7, 0.5], [91, 98, 105, 112, 180, 270, 365, 546, 728])

def ewm_features(dataframe, alphas, lags):
    if isinstance(dataframe, np.ndarray):
        dataframe = pd.DataFrame(dataframe, columns=["País", "Categoría Tarjeta", "Ventas"] + [f'feature_{i}' for i in range(3, dataframe.shape[1])])
    for alpha in alphas:
        for lag in lags:
            dataframe[f'Ventas_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'] = dataframe.groupby(
                ["País", "Categoría Tarjeta"])['Ventas'].transform(
                lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


###APLICACIÓN

# Configurar la página para que se expanda en todo el ancho
st.set_page_config(
    page_title="Aplicación de Predicciones",
    page_icon="images/growup.png",
    layout="wide"
)

st.markdown(
    f"""
    <style>
    /* Fondo de la aplicación */
    .stApp {{
        background-color: #FFFFFF;  /* Fondo blanco para mejor legibilidad */
    }}
    /* Fondo de la barra lateral */
    .css-1d391kg .css-1d391kg {{
        background-color: #FFFFFF;
    }}
    /* Títulos */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_2};  /* Títulos en azul oscuro */
    }}
    /* Texto */
    .stMarkdown, .css-1fv8s86, .css-16huue1 {{
        color: {COLOR_2};  /* Texto en azul oscuro */
    }}
    /* Etiquetas de widgets */
    .css-145kmo2 {{
        color: {COLOR_2};  /* Etiquetas en azul oscuro */
    }}
    /* Botones */
    .stButton>button {{
        background-color: {COLOR_1};  /* Botones en naranja */
        color: white;
    }}
    /* Cuadros de entrada */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        color: {COLOR_2};  /* Texto de entrada en azul oscuro */
    }}
    /* Selectores */
    .stSelectbox>div>div>div>div {{
        color: {COLOR_2};
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Título de la aplicación
st.title('Aplicación de Predicciones')

st.sidebar.image("images/growup.png", width=150)

# Crear una selección en la barra lateral para elegir el modelo
modelo_seleccionado = st.sidebar.selectbox('Seleccione el modelo de predicción:', 
                                           ['Inicio', 'Predicción de Abandono de Cliente', 'Pronóstico de Ventas'])

if modelo_seleccionado == 'Inicio':
    # Página de bienvenida
    st.image("images/portada.png", width=300)
    st.header('Bienvenido a la Aplicación de Predicciones')
    st.write('Esta aplicación le permite utilizar dos modelos de predicción:')
    st.write('1. **Predicción de Abandono de Cliente**: predice si un cliente abandonará el banco.')
    st.write('2. **Pronóstico de Ventas**: predice las ventas futuras basadas en fecha, categoría de tarjeta y país.')
    st.write('Seleccione el modelo que desea utilizar en el menú de la izquierda.')

elif modelo_seleccionado == 'Predicción de Abandono de Cliente':
    # Cargar el pipeline guardado para el modelo de abandono
    pipeline_abandono = joblib.load("models/pipeline_modelo_abandono.pkl")
    
    st.header('Predicción de Abandono de Cliente')
    
    # Crear la opción para subir un archivo
    st.header('Subir archivo de datos:')
    uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)
        
        # Mostrar las primeras filas del DataFrame
        st.subheader('Vista previa de los datos:')
        st.write(data.head())
        
        # Realizar predicciones
        predicciones = pipeline_abandono.predict(data)
        
        # Agregar las predicciones al DataFrame
        data['Predicción'] = predicciones
        data['Predicción'] = data['Predicción'].map({1: 'No Abandona', 0: 'Abandona'})
        
        # Mostrar la gráfica de predicciones usando Streamlit
        st.subheader('Gráfica de Predicciones:')

        # Obtener el conteo de las predicciones
        counts = data['Predicción'].value_counts().reset_index()
        counts.columns = ['Predicción', 'Cantidad']

        # Crear la gráfica circular con Plotly
        fig = px.pie(counts, values='Cantidad', names='Predicción', title='Distribución de Predicciones')

        # Mostrar la gráfica en Streamlit
        st.plotly_chart(fig)
        
        # Botón para descargar el archivo con predicciones
        st.subheader('Descargar tabla con predicciones:')
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label='Descargar CSV', data=csv, file_name='predicciones.csv', mime='text/csv')

    else:
        # Si no se ha subido un archivo, mostrar el formulario individual
        st.header('Ingrese la información del cliente:')
        
        col1,col2,col3,col4 = st.columns([2,1,2,2])
        
        with col1:
            edad = st.slider('Edad del Cliente', min_value=18, max_value=100, value=30)
            genero = st.selectbox('Género', ['Masculino', 'Femenino'])
            personas_cargo = st.slider('Cantidad de Personas a Cargo', min_value=0, max_value=10, value=0)
            nivel_educativo = st.selectbox('Nivel Educativo', ['Secundaria', 'Graduado', 'Sin Educación',
                                                            'Desconocido', 'Universidad', 'Postgrado', 'Doctorado'])
            estado_civil = st.selectbox('Estado Civil', ['Casado', 'Soltero', 'Desconocido'])
            rango_ingresos = st.selectbox('Rango de Ingresos', ['$60K - $80K', 'Menos de $40K', '$80K - $120K',
                                                                '$40K - $60K', 'Más de $120K', 'Desconocido'])
            categoria_tarjeta = st.selectbox('Categoría Tarjeta', ['Azul', 'Plata', 'Oro', 'Platino'])
            antiguedad_cuenta = st.number_input('Antigüedad de la Cuenta (meses)', min_value=1, max_value=120, value=12)
        
        with col3:
            total_productos_bancarios = st.number_input('Total de Productos Bancarios', min_value=1, max_value=10, value=3)
            meses_inactivos = st.number_input('Meses Inactivos en el Último Año', min_value=0, max_value=12, value=1)
            frecuencia_contacto = st.number_input('Frecuencia de Contacto con el Banco', min_value=0, max_value=10, value=2)
            limite_credito = st.number_input('Límite de Crédito', min_value=0.0, value=5000.0)
            saldo_pendiente = st.number_input('Saldo Pendiente de Tarjeta', min_value=0.0, value=1000.0)
            promedio_credito = st.number_input('Promedio de Crédito Disponible', min_value=0.0, value=4000.0)
            monto_transacciones = st.number_input('Monto Total de Transacciones', min_value=0.0, value=500.0)
            numero_transacciones = st.number_input('Número Total de Transacciones', min_value=0, value=20)
            porcentaje_uso_credito = st.number_input('Porcentaje de Uso de Crédito', min_value=0.0, value=0.5)
        
        with col4:
        # Botón de predicción
            if st.button('Predecir'):
                # Crear un dataframe con los valores ingresados
                datos_cliente = pd.DataFrame({
                    'Edad Cliente': [edad],
                    'Genero': [genero],
                    'Cant. Personas a Cargo': [personas_cargo],
                    'Nivel Educativo': [nivel_educativo],
                    'Estado Civil': [estado_civil],
                    'Rango Ingresos': [rango_ingresos],
                    'Categoria Tarjeta': [categoria_tarjeta],
                    'Antiguedad Cuenta': [antiguedad_cuenta],
                    'Total Productos Bancarios': [total_productos_bancarios],
                    'Meses Inactivos Ultimo Año': [meses_inactivos],
                    'Frecuencia de Contacto al Banco': [frecuencia_contacto],
                    'Limite Credito': [limite_credito],
                    'Saldo Pendiente Tarjeta': [saldo_pendiente],
                    'Promedio Credito Disponible': [promedio_credito],
                    'Monto Total Transacciones': [monto_transacciones],
                    'Numero Total Transacciones': [numero_transacciones],
                    'Porcentaje Uso Credito': [porcentaje_uso_credito]
                })
            
                # Realizar la predicción
                prediccion = pipeline_abandono.predict(datos_cliente)
            
                # Mostrar el resultado
                if prediccion[0] == 1:
                    st.success('El cliente **NO abandona** el banco.')
                else:
                    st.warning('El cliente **abandona** el banco.')


elif modelo_seleccionado == 'Pronóstico de Ventas':
    # Cargar el pipeline guardado para el modelo de ventas
    pipeline_ventas = joblib.load('models/pipeline_model.pkl')
    
    
    st.header('Pronóstico de Ventas')
    
    st.subheader('Seleccione los parámetros para el pronóstico:')
    
    # Crear columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Selección de rango de fechas
        fecha_inicio = st.date_input('Fecha de inicio', value=pd.to_datetime('2024-01-01'), min_value=pd.to_datetime('2024-01-01'))
        fecha_fin = st.date_input('Fecha de fin', value=pd.to_datetime('2024-02-28'), min_value=fecha_inicio)
    
    with col2:
        # Selección múltiple de categorías de tarjeta
        opciones_categorias = ['Azul', 'Plata', 'Oro', 'Platino']
        categorias_tarjeta = st.multiselect('Categorías de Tarjeta', opciones_categorias, default=['Oro'])
    
    with col3:
        # Selección múltiple de países
        opciones_paises = ['Argentina', 'Chile', 'Colombia', 'Costa Rica', 'Ecuador', 'El Salvador', 'Estados Unidos', 'Guatemala', 'Perú', 'Uruguay']
        paises_seleccionados = st.multiselect('Países', opciones_paises, default=['Costa Rica'])
    
    # Verificar que se haya seleccionado al menos una categoría y un país
    if not categorias_tarjeta:
        st.warning('Por favor, seleccione al menos una Categoría de Tarjeta.')
    elif not paises_seleccionados:
        st.warning('Por favor, seleccione al menos un País.')
    else:
        # Botón para generar el pronóstico
        if st.button('Generar Pronóstico'):
            # Generar rango de fechas
            fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
            
            # Crear todas las combinaciones posibles
            combinaciones = list(product(fechas, categorias_tarjeta, paises_seleccionados))
            
            # Crear DataFrame con las combinaciones
            datos_ventas = pd.DataFrame(combinaciones, columns=['Fecha Pedido', 'Categoría Tarjeta', 'País'])
            
            # Mapear las categorías de tarjeta y países a códigos numéricos usando LabelEncoder
            le_categoria = LabelEncoder()
            le_pais = LabelEncoder()
            
            # Ajustar los LabelEncoders con las categorías y países originales
            categorias_originales = ['Azul', 'Plata', 'Oro', 'Platino']
            paises_originales = ['Argentina', 'Chile', 'Colombia', 'Costa Rica', 'Ecuador', 'El Salvador', 'Estados Unidos', 'Guatemala', 'Perú', 'Uruguay']
            
            le_categoria.fit(categorias_originales)
            le_pais.fit(paises_originales)
            
            # Verificar que las categorías y países seleccionados estén en las categorías originales
            if not set(categorias_tarjeta).issubset(set(categorias_originales)):
                st.error("Ha seleccionado categorías de tarjeta que no están en el modelo entrenado.")
            elif not set(paises_seleccionados).issubset(set(paises_originales)):
                st.error("Ha seleccionado países que no están en el modelo entrenado.")
            else:
                # Transformar las categorías en los nuevos datos
                datos_ventas['Categoría Tarjeta'] = le_categoria.transform(datos_ventas['Categoría Tarjeta'])+1
                datos_ventas['País'] = le_pais.transform(datos_ventas['País'])+1
                
                # Agregar la columna 'Ventas' con valores NaN
                datos_ventas['Ventas'] = np.nan  # No conocemos las ventas aún
                
                # Aplicar manualmente las transformaciones de fecha
                datos_ventas = create_date_features_manually(datos_ventas)
                
                # Eliminar la columna 'Fecha Pedido' y 'Ventas' antes de pasar por el pipeline
                X_nuevos = datos_ventas.drop(columns=['Fecha Pedido', 'Ventas'])
                
                # Realizar la predicción
                predicciones = pipeline_ventas.predict(X_nuevos)
                
                # Aplicar la inversa de la transformación logarítmica a las predicciones
                predicciones = np.expm1(predicciones)  # Invertir log(1 + ventas)
                
                # Agregar las predicciones al DataFrame original
                datos_ventas['Ventas'] = predicciones
                
                # Mapear de vuelta los códigos numéricos a los nombres originales
                datos_ventas['Categoría Tarjeta'] = datos_ventas['Categoría Tarjeta'] - 1
                datos_ventas['País'] = datos_ventas['País'] - 1
                datos_ventas['Categoría Tarjeta'] = le_categoria.inverse_transform(datos_ventas['Categoría Tarjeta'])
                datos_ventas['País'] = le_pais.inverse_transform(datos_ventas['País'])
                
                # Mantener solo las columnas requeridas: Fecha Pedido, País, Categoría Tarjeta, Ventas
                resultado = datos_ventas[['Fecha Pedido', 'País', 'Categoría Tarjeta', 'Ventas']]
                
                # Mostrar el resultado
                st.subheader('Resultados de la Predicción')
                st.write(resultado.head())
                
                # Mostrar el gráfico de pronóstico de ventas
                st.subheader('Gráfico de Pronóstico de Ventas')
                
                # Agrupar los datos por 'Fecha Pedido', 'Categoría Tarjeta' y 'País'
                ventas_agrupadas = resultado.groupby(['Fecha Pedido', 'Categoría Tarjeta', 'País'])['Ventas'].sum().reset_index()
                
                # Crear una columna que combine 'Categoría Tarjeta' y 'País' para las etiquetas
                ventas_agrupadas['Categoría_País'] = ventas_agrupadas['Categoría Tarjeta'] + ' - ' + ventas_agrupadas['País']
                
                # Crear la gráfica con Plotly
                fig = px.line(ventas_agrupadas, x='Fecha Pedido', y='Ventas', color='Categoría_País',
                              title='Pronóstico de Ventas por Categoría y País')
                
                # Mostrar la gráfica en Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
                # Botón para descargar el archivo con predicciones
                st.subheader('Descargar resultados')
                csv = resultado.to_csv(index=False).encode('utf-8')
                st.download_button(label='Descargar CSV', data=csv, file_name='pronostico_ventas.csv', mime='text/csv')
