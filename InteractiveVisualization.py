import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

# Agregar selección de rango de fechas
st.sidebar.title("Filtrar por rango de fechas")
min_fecha = precios['FECHA'].min().date()  # Convertir a date
max_fecha = precios['FECHA'].max().date()  # Convertir a date

# Agregar el slider para la selección de fechas
rango_fechas = st.sidebar.slider("Seleccione el rango de fechas", min_value=min_fecha, max_value=max_fecha, value=(min_fecha, max_fecha))

# Convertir las fechas seleccionadas a Timestamps para poder filtrar
rango_fechas = pd.to_datetime(rango_fechas)

# Filtrar los datos con el rango de fechas seleccionado
precios_seleccionado = precios_seleccionado[(precios_seleccionado['FECHA'] >= rango_fechas[0]) & (precios_seleccionado['FECHA'] <= rango_fechas[1])]
consumo_seleccionado = consumo_seleccionado[(consumo_seleccionado['Fecha'] >= rango_fechas[0]) & (consumo_seleccionado['Fecha'] <= rango_fechas[1])]
importacion_seleccionado = importacion_seleccionado[(importacion_seleccionado['Fecha'] >= rango_fechas[0]) & (importacion_seleccionado['Fecha'] <= rango_fechas[1])]

# Gráficos interactivos con paleta de colores
if visualizacion == "Precios":
    st.subheader("Evolución de Precios de Combustibles")
    fig = px.line(precios_seleccionado, x='FECHA', y=['Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.'],
                  title="Precios de Combustibles a lo Largo del Tiempo", 
                  color_discrete_sequence=['#1f77b4'])  # Azul Honolulu para estabilidad y confianza en precios
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio (Q)")
    st.plotly_chart(fig)
    
elif visualizacion == "Consumo":
    st.subheader("Consumo de Combustibles")
    fig = px.line(consumo_seleccionado, x='Fecha', y=['Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petróleo'],
                  title="Consumo de Combustibles a lo Largo del Tiempo", 
                  color_discrete_sequence=['#2ca02c'])  # Pigmento Verde para sostenibilidad en consumo
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Consumo (m³)")
    st.plotly_chart(fig)

elif visualizacion == "Importación":
    st.subheader("Importación de Combustibles")
    fig = px.line(importacion_seleccionado, x='Fecha', y=['Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petróleo'],
                  title="Importación de Combustibles a lo Largo del Tiempo", 
                  color_discrete_sequence=['#ff7f0e'])  # Naranja para energía y dinamismo en importaciones
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Volumen Importado (m³)")
    st.plotly_chart(fig)

# Predecir precios usando 3 modelos
st.sidebar.title("Predicción de Precios")
if st.sidebar.checkbox("Mostrar Predicción de Precios"):
    st.subheader("Modelos de Predicción")

    # Definir datos para el modelo
    precios_seleccionado['FECHA_ordinal'] = precios_seleccionado['FECHA'].map(pd.Timestamp.toordinal)
    X = precios_seleccionado[['FECHA_ordinal']]
    y = precios_seleccionado['Regular']  # Precio de gasolina regular como ejemplo
    
    # Separar datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo 1: Regresión Lineal
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    error_mae_lr = mean_absolute_error(y_test, y_pred_lr)
    
    # Modelo 2: Árbol de Decisión
    modelo_tree = DecisionTreeRegressor(random_state=42)
    modelo_tree.fit(X_train, y_train)
    y_pred_tree = modelo_tree.predict(X_test)
    error_mae_tree = mean_absolute_error(y_test, y_pred_tree)

    # Modelo 3: Random Forest
    modelo_rf = RandomForestRegressor(random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    error_mae_rf = mean_absolute_error(y_test, y_pred_rf)

    # Mostrar el desempeño de los tres modelos
    st.write(f"**Error absoluto medio (MAE) de Regresión Lineal**: {error_mae_lr:.2f}")
    st.write(f"**Error absoluto medio (MAE) de Árbol de Decisión**: {error_mae_tree:.2f}")
    st.write(f"**Error absoluto medio (MAE) de Random Forest**: {error_mae_rf:.2f}")

    # Tabla comparativa de errores
    comparacion_modelos = pd.DataFrame({
        'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest'],
        'MAE': [error_mae_lr, error_mae_tree, error_mae_rf]
    })
    st.subheader("Comparativa de Modelos")
    st.dataframe(comparacion_modelos)

    # Visualización de predicciones del modelo de Regresión Lineal
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['FECHA_ordinal'], y_test, color='#1f77b4', label='Datos reales')  # Azul Honolulu
    plt.plot(X_test['FECHA_ordinal'], y_pred_lr, color='#d62728', label='Predicciones (Regresión Lineal)', linestyle='--')  # Rojo Bombero
    plt.xlabel('Fecha (Ordinal)')
    plt.ylabel('Precio de Gasolina Regular')
    plt.title('Predicciones vs Datos Reales (Regresión Lineal)')
    plt.legend()
    st.pyplot(plt)

    # Visualización de predicciones del modelo de Árbol de Decisión
    plt.figure(figsize=(10, 6))
    # Ordenamos los datos según el valor de 'FECHA_ordinal'
    sorted_idx_tree = X_test['FECHA_ordinal'].argsort()

    # Graficar los puntos reales
    plt.scatter(X_test['FECHA_ordinal'], y_test, color='#1f77b4', label='Datos reales')  # Azul Honolulu

    # Graficar la línea de regresión, ordenando las predicciones según la fecha
    plt.plot(X_test['FECHA_ordinal'].iloc[sorted_idx_tree], y_pred_tree[sorted_idx_tree], color='#2ca02c', linewidth=2, label='Línea de Predicción (Árbol de Decisión)', linestyle='--')  # Pigmento Verde

    plt.xlabel('Fecha (Ordinal)')
    plt.ylabel('Precio de Gasolina Regular')
    plt.title('Predicciones vs Datos Reales (Árbol de Decisión)')
    plt.legend()
    st.pyplot(plt)

    # Visualización de predicciones del modelo de Random Forest
    plt.figure(figsize=(10, 6))
    # Ordenamos los datos según el valor de 'FECHA_ordinal'
    sorted_idx_rf = X_test['FECHA_ordinal'].argsort()

    # Graficar los puntos reales
    plt.scatter(X_test['FECHA_ordinal'], y_test, color='#1f77b4', label='Datos reales')  # Azul Honolulu

    # Graficar la línea de regresión, ordenando las predicciones según la fecha
    plt.plot(X_test['FECHA_ordinal'].iloc[sorted_idx_rf], y_pred_rf[sorted_idx_rf], color='#bcbd22', linewidth=2, label='Línea de Predicción (Random Forest)', linestyle='--')  # Oro Viejo

    plt.xlabel('Fecha (Ordinal)')
    plt.ylabel('Precio de Gasolina Regular')
    plt.title('Predicciones vs Datos Reales (Random Forest)')
    plt.legend()
    st.pyplot(plt)

# Footer
st.sidebar.markdown("**Hecho por: Daniel Valdez y Emilio Solano**")
