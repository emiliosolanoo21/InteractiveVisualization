# Interactive Fuel Dashboard

This project is an interactive tool for visualizing and predicting fuel prices, developed with **Streamlit**. It uses historical data on fuel consumption, importation, and prices in Guatemala, with advanced options to visualize trends and perform predictions using Machine Learning algorithms.

## Features

- Visualization of **prices, consumption, and importation** of fuels such as Regular Gasoline, Premium Gasoline, Diesel, and LPG.
- **Filtering by date range** and detail level: monthly, quarterly, and annual.
- Interactive charts with different options: **lines** and **bars**.
- Prediction models implemented:
  - **Linear Regression**
  - **Decision Trees**
  - **Random Forest**
- Model error comparison: MAE, MSE, RMSE, and R² Score.
- **Advanced charts**: stacked area, scatter plot, and heatmap.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/emiliosolanoo21/InteractiveVisualization.git
   cd InteractiveVisualization
   ```
2. Run the application in Console:
    ```bash
   streamlit run InteractiveVisualization.py
    ```
## Data
The data used for visualization and prediction is divided into several Excel files:

- PRICES 2021.xlsx, PRICES 2022.xlsx, PRICES 2023.xlsx, PRICES 2024.xlsx: contain historical fuel prices.
- CONSUMPTION.xlsx: monthly fuel consumption data.
- IMPORTATION.xlsx: monthly fuel importation data.

## Design Decisions
### Color Palette
A color palette was chosen to facilitate the differentiation between fuel types and ensure good visibility in interactive charts:
- **Regular Gasoline**: Honolulu Blue (#1f77b4)
- **Premium Gasoline**: Orange (Wheel) (#ff7f0e)
- **Diesel**: Pigment Green (#2ca02c)
- **LPG**: Fire Engine Red (#d62728)

The choice is based on colors that are accessible for people with color blindness and maintain good contrast between them.

## Prediction Models
To predict fuel prices, the following models were selected due to their ability to handle time series data:

- **Linear Regression**: A simple and easy-to-interpret model.
- **Decision Tree**: Good fit for nonlinear data.
- **Random Forest**: A robust model with low variance.

**MAE** (Mean Absolute Error) was used to compare the model results and select the most suitable one.

## Authors
- Daniel Valdez
- Emilio Solano
