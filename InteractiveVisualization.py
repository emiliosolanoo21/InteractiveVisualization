import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar los datos
precios_2021 = pd.read_excel('PRECIOS 2021.xlsx', engine='openpyxl')
precios_2022 = pd.read_excel('PRECIOS 2022.xlsx', engine='openpyxl')
precios_2023 = pd.read_excel('PRECIOS 2023.xlsx', engine='openpyxl')
precios_2024 = pd.read_excel('PRECIOS 2024.xlsx', engine='openpyxl')

consumo = pd.read_excel('CONSUMO.xlsx', engine='openpyxl')
importacion = pd.read_excel('IMPORTACION.xlsx', engine='openpyxl')

# Unificar datos de precios
precios = pd.concat([precios_2021, precios_2022, precios_2023, precios_2024], ignore_index=True)

# Convertir columnas de fecha a datetime
consumo['Fecha'] = pd.to_datetime(consumo['Fecha'], format='%Y-%m')
importacion['Fecha'] = pd.to_datetime(importacion['Fecha'], format='%Y-%m')
precios['FECHA'] = pd.to_datetime(precios['FECHA'], format='%Y-%m-%d')

consumo_seleccionado = consumo[['Fecha', 'Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petróleo']]
importacion_seleccionado = importacion[['Fecha', 'Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petróleo']]
precios_seleccionado = precios[['FECHA', 'Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.']]

# Configurar la página
st.set_page_config(page_title="Dashboard de Combustibles", layout="wide")

# Título
st.title("Dashboard Interactivo de Combustibles")

# Sidebar para seleccionar qué tipo de datos ver
st.sidebar.title("Opciones de Visualización")
visualizacion = st.sidebar.selectbox("Seleccione qué visualizar", ["Precios", "Consumo", "Importación"])

# Gráficos interactivos
if visualizacion == "Precios":
    st.subheader("Evolución de Precios de Combustibles")
    fig = px.line(precios_seleccionado, x='FECHA', y=['Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.'],
                  title="Precios de Combustibles a lo Largo del Tiempo")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio (Q)")
    st.plotly_chart(fig)
    
elif visualizacion == "Consumo":
    st.subheader("Consumo de Combustibles")
    fig = px.line(consumo_seleccionado, x='Fecha', y=['Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petróleo'],
                  title="Consumo de Combustibles a lo Largo del Tiempo")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Consumo (m³)")
    st.plotly_chart(fig)

elif visualizacion == "Importación":
    st.subheader("Importación de Combustibles")
    fig = px.line(importacion_seleccionado, x='Fecha', y=['Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petróleo'],
                  title="Importación de Combustibles a lo Largo del Tiempo")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Volumen Importado (m³)")
    st.plotly_chart(fig)

# Predecir precios usando un modelo simple de regresión lineal
st.sidebar.title("Predicción de Precios")
if st.sidebar.checkbox("Mostrar Predicción de Precios"):
    st.subheader("Modelo de Predicción: Regresión Lineal")
    
    # Definir datos para el modelo
    precios_seleccionado['FECHA_ordinal'] = precios_seleccionado['FECHA'].map(pd.Timestamp.toordinal)
    X = precios_seleccionado[['FECHA_ordinal']]
    y = precios_seleccionado['Regular']  # Precio de gasolina regular como ejemplo
    
    # Separar datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear el modelo y entrenarlo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = modelo.predict(X_test)
    
    # Mostrar el desempeño del modelo
    error = mean_squared_error(y_test, y_pred)
    st.write(f"Error cuadrático medio (MSE) del modelo: {error}")
    
    # Visualizar las predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Datos reales')
    plt.plot(X_test, y_pred, color='red', label='Predicciones')
    plt.xlabel('Fecha (Ordinal)')
    plt.ylabel('Precio de Gasolina Regular')
    plt.title('Predicciones vs Datos Reales')
    plt.legend()
    st.pyplot(plt)

# Footer
st.sidebar.markdown("**Hecho por: Daniel Valdez y Emilio Solano**")
