\# CRAVE



\## Crime Risk Analytics and Forecasting Engine



CRAVE is a Python-based analytics and machine-learning project for studying patterns in synthetic crime incident data, identifying geographic risk cells, and generating short-term statistical forecasts.



The project is designed as a research and engineering prototype using synthetic data. It supports experimentation with data engineering, feature engineering, machine learning, statistical forecasting, and analytical reporting.



\## Features



\- Synthetic crime incident dataset

\- CSV data loading and validation

\- Timestamp and geographic feature engineering

\- Geographic risk-cell aggregation

\- Crime incident frequency analysis

\- Random Forest regression model

\- MAE and RMSE model evaluation

\- Seven-day statistical forecasting

\- LOW, MEDIUM, and HIGH risk classification

\- Command-line interface

\- Automated testing with pytest



\## Project structure



```text

CRAVE\_Crime\_Risk\_Analytics\_and\_Forecasting\_Engine/

│

├── data/

│   └── sample\_crime\_data.csv

│

├── docs/

│

├── src/

│   └── crave/

│       ├── \_\_init\_\_.py

│       ├── cli.py

│       ├── data.py

│       ├── features.py

│       ├── forecast.py

│       └── model.py

│

├── tests/

│   └── test\_crave.py

│

├── main.py

├── requirements.txt

└── README.md

