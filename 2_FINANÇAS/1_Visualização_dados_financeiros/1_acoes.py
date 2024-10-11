import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from prophet import Prophet
from pandas_datareader import data
import yfinance as yf
import os


azul_df = yf.download('AZUL4.SA', start='2015-01-01')
csv_file = 'azul4.csv'
print(azul_df.head())

if not os.path.exists(csv_file):
    azul_df.to_csv(csv_file)
    print(f"Arquivo '{csv_file}' salvo com sucesso!")
else:
    print(f"O arquivo '{csv_file}' já existe. Nenhuma ação realizada.")

azul_df.info()



azul_df.describe()



#Média de 26.527079 com variação de 12.976454 pra cima ou para baixo

maior_fechamento = azul_df[azul_df['Close'] >= 62.4]
menor_fechamento = azul_df[azul_df['Close']<= 4.04]


