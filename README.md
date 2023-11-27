# Data-Engineer

## Loader
### 01_data2db
- Open API / Close API
- to MYSQL DB

![image](./README_ASSETS/01_data2db.png)

### 02_db2local
- MYSQL DB -> to local

![image](./README_ASSETS/02_db2local.png)

## Model
### model_parameter
- Fundamental_Param
- Technical_Param 
    - Moving_Average_Param
    - Linear_Coef_Param

![image](./README_ASSETS/03_model_parameter.png)

### 04_model
- Fundamental_Model
- Technical_Model
- Trader_Model

![image](./README_ASSETS/04_model.png)

### 05_main (model + model_parameter)
- Fundamental
    - Fundamental_Param
    - Fundamental_Model
- Technical
    - Technical_Param
    - Technical_Model
- Trader_Model

![image](./README_ASSETS/05_main.png)


## Order
### Order
#### 06_order_ki
- KoreaInvestment

## 99
### Analysis
### 99_factor_analysis
- Model Paramter Tuning Related

![image](./README_ASSETS/fundamental_analysis_fig.png)
![image](./README_ASSETS/technical_analysis_fig.png)

### 99_market_analysis
- Market Trader Volume Related
    - [Corp / Foriegn / Indivisual]

![image](./README_ASSETS/market_analysis_fig.png)

### 99_position_analysis
- Current Position Related

![image](./README_ASSETS/position_trader_fig.png)
![image](./README_ASSETS/position_price_fig.png)

### 99_stock_analysis
- Main result analysis before order

![image](./README_ASSETS/stock_analysis.png)