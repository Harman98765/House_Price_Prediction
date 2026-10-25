# House Price Prediction

## Overview
This project uses Machine Learning to predict real estate prices using Multiple Linear Regression. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a Streamlit web application for interactive predictions.

## Dataset
The project uses the [UCI Real Estate Valuation Dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set), containing 414 property transactions, six input features, and one target variable.

**Input Features:**
- Transaction date
- House age
- Distance to the nearest MRT station
- Number of convenience stores
- Latitude
- Longitude

**Target Variable:** House price per unit area.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

## Project Structure
```text
house-price-prediction/
├── house_price_prediction.ipynb
├── app.py
├── requirements.txt
├── Real estate valuation data set.xlsx
└── README.md
```

## Methodology
1. **Dataset Handling:** Import and inspect the dataset, including its dimensions, columns, and data types.
2. **Data Preprocessing:** Check missing values, duplicates, and potential outliers; remove the transaction ID and scale input features.
3. **Exploratory Data Analysis:** Calculate descriptive statistics and visualize the data using a histogram and correlation heatmap.
4. **Model Training:** Train a Multiple Linear Regression model.
5. **Evaluation:** Evaluate performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
6. **Web Application:** Use Streamlit to provide an interface for entering property details and obtaining price predictions.

## Live Demo
**Streamlit App:** https://housepricepredictionproject246.streamlit.app/


## Application Preview

![House Price Prediction App](screenshots/app_screenshot.png)


## Evaluation Metrics
- **MSE:** Measures the average squared prediction error.
- **RMSE:** Measures prediction error in the target variable's units.
- **R²:** Measures how much variation in the target variable is explained by the model.

## Limitations
Predictions are estimates based on the dataset and trained regression model. The target represents price per unit area, not the total price of a property. Model performance may be limited when predicting properties outside the dataset's geographic area or distribution.

## Acknowledgements
Dataset source: [UCI Machine Learning Repository — Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set).
