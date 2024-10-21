import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# **Nivel de detalle y tipo de gráfico**: Selector de nivel de agregación y tipo de gráfico para las visualizaciones interactivas
st.sidebar.title("Opciones de Nivel de Detalle y Tipo de Gráfico")
nivel_agrupacion = st.sidebar.selectbox("Seleccione el nivel de agrupación", ["Mes", "Trimestre", "Año"])
tipo_grafico = st.sidebar.selectbox("Seleccione el tipo de gráfico", ["Líneas", "Barras"])

# Función para agregar datos por nivel de detalle
def agregar_datos(df, nivel_agrupacion):
    if nivel_agrupacion == "Mes":
        return df.resample('M', on='FECHA').mean().reset_index()
    elif nivel_agrupacion == "Trimestre":
        return df.resample('Q', on='FECHA').mean().reset_index()
    elif nivel_agrupacion == "Año":
        return df.resample('Y', on='FECHA').mean().reset_index()

# Aplicar la agrupación seleccionada a los datos (esto no afecta las gráficas de predicción)
precios_agrupado = agregar_datos(precios_seleccionado, nivel_agrupacion)
consumo_agrupado = agregar_datos(consumo_seleccionado.rename(columns={"Fecha": "FECHA"}), nivel_agrupacion).rename(columns={"FECHA": "Fecha"})
importacion_agrupado = agregar_datos(importacion_seleccionado.rename(columns={"Fecha": "FECHA"}), nivel_agrupacion).rename(columns={"FECHA": "Fecha"})

# **Gráficos enlazados**: Agregar un selector de tipo de combustible
combustible_seleccionado = st.sidebar.selectbox("Seleccione el tipo de combustible", ["Gasolina regular", "Gasolina superior", "Diesel bajo azufre", "Gas licuado de petróleo"])
typesCombustibles = {
    "Gasolina regular": "Regular",
    "Gasolina superior": "Superior",
    "Diesel bajo azufre": "Diesel",
    "Gas licuado de petróleo": "Glp Cilindro 25Lbs."   
}

# Función para generar gráficos con base en el tipo de gráfico seleccionado
def generar_grafico(df, x_col, y_col, tipo_grafico, titulo, xaxis_title, yaxis_title, color_sequence):
    if tipo_grafico == "Líneas":
        fig = px.line(df, x=x_col, y=y_col, title=titulo, color_discrete_sequence=color_sequence)
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=titulo, color_discrete_sequence=color_sequence)
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig

# Gráfico de precios (general, no enlazado)
if visualizacion == "Precios":
    st.subheader("Evolución de Precios de Combustibles")
    fig = generar_grafico(precios_agrupado, 'FECHA', ['Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.'],
                          tipo_grafico, "Precios de Combustibles a lo Largo del Tiempo", "Fecha", "Precio (Q)", 
                          ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    st.plotly_chart(fig)

# Gráficos de consumo e importación, enlazados con el combustible seleccionado
if visualizacion == "Consumo":
    st.subheader(f"Consumo de {combustible_seleccionado}")
    fig = generar_grafico(consumo_agrupado, 'Fecha', combustible_seleccionado,
                          tipo_grafico, f"Consumo de {combustible_seleccionado} a lo Largo del Tiempo", "Fecha", "Consumo (m³)", 
                          ['#1f77b4'])
    st.plotly_chart(fig)

elif visualizacion == "Importación":
    st.subheader(f"Importación de {combustible_seleccionado}")
    fig = generar_grafico(importacion_agrupado, 'Fecha', combustible_seleccionado,
                          tipo_grafico, f"Importación de {combustible_seleccionado} a lo Largo del Tiempo", "Fecha", "Volumen Importado (m³)", 
                          ['#ff7f0e'])
    st.plotly_chart(fig)

if st.sidebar.checkbox("Mostar Visualizaciones Avanzadas"):
    # Gráfico de área acumulada de precios por combustible
    st.subheader("Evolución Acumulada de Precios de Combustibles")
    fig_area = px.area(precios_agrupado, x='FECHA', y=['Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.'],
                    title="Evolución Acumulada de Precios de Combustibles", 
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    fig_area.update_layout(xaxis_title="Fecha", yaxis_title="Precio Acumulado (Q)")
    st.plotly_chart(fig_area)


    # Gráfico de dispersión de importación vs consumo para el combustible seleccionado
    st.subheader(f"Relación entre Importación y Consumo de {combustible_seleccionado}")
    fig_dispersion = px.scatter(x=importacion_agrupado['Fecha'], y=importacion_agrupado[combustible_seleccionado],
                                size=consumo_agrupado[combustible_seleccionado], 
                                title=f"Importación vs Consumo de {combustible_seleccionado}",
                                labels={'x': 'Fecha', 'y': f'Importación de {combustible_seleccionado} (m³)'})
    fig_dispersion.update_layout(xaxis_title="Fecha", yaxis_title=f"Importación de {combustible_seleccionado} (m³)")
    st.plotly_chart(fig_dispersion)


    # Mapa de calor de precios de combustibles
    st.subheader("Mapa de Calor de Precios por Combustible")
    fig_heatmap = px.imshow(precios_agrupado[['FECHA', 'Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.']].set_index('FECHA').T,
                            title="Mapa de Calor de Precios por Combustible a lo Largo del Tiempo",
                            labels={'x': 'Fecha', 'y': 'Combustible'})
    fig_heatmap.update_layout(xaxis_title="Fecha", yaxis_title="Tipo de Combustible")
    st.plotly_chart(fig_heatmap)


    # Gráfico de líneas del porcentaje de cambio mensual en los precios de combustibles
    st.subheader("Porcentaje de Cambio Mensual de Precios de Combustibles")
    precios_agrupado_pct_change = precios_agrupado.set_index('FECHA').pct_change().reset_index()
    fig_pct_change = px.line(precios_agrupado_pct_change, x='FECHA', 
                            y=['Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.'],
                            title="Porcentaje de Cambio Mensual de Precios de Combustibles",
                            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    fig_pct_change.update_layout(xaxis_title="Fecha", yaxis_title="Porcentaje de Cambio (%)")
    st.plotly_chart(fig_pct_change)


st.sidebar.title("Predicción de Precios")
if st.sidebar.checkbox("Mostrar Predicción de Precios"):
    st.subheader("Modelos de Predicción")

    # Definir datos para el modelo (sin agregar por nivel de detalle)
    precios_seleccionado['FECHA_ordinal'] = precios_seleccionado['FECHA'].map(pd.Timestamp.toordinal)
    X = precios_seleccionado[['FECHA_ordinal']]
    y = precios_seleccionado[typesCombustibles[combustible_seleccionado]]
    
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
    # Tabla comparativa de errores
    
    # Modelo 1: Regresión Lineal
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    error_mae_lr = mean_absolute_error(y_test, y_pred_lr)
    error_mse_lr = mean_squared_error(y_test, y_pred_lr)
    error_rmse_lr = error_mse_lr ** 0.5
    r2_lr = modelo_lr.score(X_test, y_test)
    
    # Modelo 2: Árbol de Decisión
    modelo_tree = DecisionTreeRegressor(random_state=42)
    modelo_tree.fit(X_train, y_train)
    y_pred_tree = modelo_tree.predict(X_test)
    error_mae_tree = mean_absolute_error(y_test, y_pred_tree)
    error_mse_tree = mean_squared_error(y_test, y_pred_tree)
    error_rmse_tree = error_mse_tree ** 0.5
    r2_tree = modelo_tree.score(X_test, y_test)

    # Modelo 3: Random Forest
    modelo_rf = RandomForestRegressor(random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    error_mae_rf = mean_absolute_error(y_test, y_pred_rf)
    error_mse_rf = mean_squared_error(y_test, y_pred_rf)
    error_rmse_rf = error_mse_rf ** 0.5
    r2_rf = modelo_rf.score(X_test, y_test)

   # Mostrar resultados de los modelos
    st.write("**Error absoluto medio (MAE) de los Modelos:**")
    st.write(f"Regresión Lineal: {error_mae_lr:.4f}")
    st.write(f"Árbol de Decisión: {error_mae_tree:.4f}")
    st.write(f"Random Forest: {error_mae_rf:.4f}")

    # Visualizar predicciones del mejor modelo
    mejor_modelo = min([(modelo_lr, y_pred_lr, error_mae_lr),
                        (modelo_tree, y_pred_tree, error_mae_tree),
                        (modelo_rf, y_pred_rf, error_mae_rf)],
                       key=lambda x: x[2])[0]

    # Predecir valores para todo el conjunto de datos
    precios_seleccionado['Prediccion'] = mejor_modelo.predict(precios_seleccionado[['FECHA_ordinal']])

    # Graficar las predicciones junto a los valores reales
    fig_pred = px.line(precios_seleccionado, x='FECHA', y=[typesCombustibles[combustible_seleccionado], 'Prediccion'],
                       title=f"Predicción de {combustible_seleccionado} vs Valores Reales",
                       labels={'value': 'Precio (Q)', 'FECHA': 'Fecha'},
                       color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    fig_pred.update_layout(xaxis_title="Fecha", yaxis_title="Precio (Q)")
    st.plotly_chart(fig_pred)

    # Tabla comparativa detallada
    comparacion_modelos = pd.DataFrame({
        'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest'],
        'MAE': [error_mae_lr, error_mae_tree, error_mae_rf],
        'MSE': [error_mse_lr, error_mse_tree, error_mse_rf],
        'RMSE': [error_rmse_lr, error_rmse_tree, error_rmse_rf],
        'R² Score': [r2_lr, r2_tree, r2_rf]
    })

    # Permitir al usuario seleccionar qué modelos comparar
    st.sidebar.title("Seleccionar Modelos para Comparar")
    modelos_a_comparar = st.sidebar.multiselect(
        "Seleccione los modelos que desea comparar",
        options=['Regresión Lineal', 'Árbol de Decisión', 'Random Forest'],
        default=['Regresión Lineal', 'Árbol de Decisión', 'Random Forest']
    )

    # Filtrar la tabla comparativa basada en la selección del usuario
    comparacion_modelos_filtrada = comparacion_modelos[comparacion_modelos['Modelo'].isin(modelos_a_comparar)]

    # Mostrar la tabla comparativa filtrada
    st.subheader("Comparativa Detallada de Modelos Seleccionados")
    st.dataframe(comparacion_modelos_filtrada)

    # Visualización de predicciones del modelo de Regresión Lineal
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['FECHA_ordinal'], y_test, color='#1f77b4', label='Datos reales')  # Azul Honolulu
    plt.plot(X_test['FECHA_ordinal'], y_pred_lr, color='#d62728', label='Predicciones (Regresión Lineal)', linestyle='--')  # Rojo Bombero
    plt.xlabel('Fecha (Ordinal)')
    plt.ylabel(f'Precio de {combustible_seleccionado}')
    plt.title(f'Predicciones vs Datos Reales (Regresión Lineal) para {combustible_seleccionado}')
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
    plt.ylabel(f'Precio de {combustible_seleccionado}')
    plt.title(f'Predicciones vs Datos Reales (Árbol de Decisión) para {combustible_seleccionado}')
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
    plt.ylabel(f'Precio de {combustible_seleccionado}')
    plt.title(f'Predicciones vs Datos Reales (Random Forest) para {combustible_seleccionado}')
    plt.legend()
    st.pyplot(plt)

# Footer
st.sidebar.markdown("**Hecho por: Daniel Valdez y Emilio Solano**")
