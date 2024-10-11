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



# Lista de ações
acoes = ['GOLL4.SA', 'AZUL4.SA', 'CVCB3.SA', 'WEGE3.SA', 'MGLU3.SA', 'TOTS3.SA', 'BOVA11.SA']

# Criar um DataFrame vazio para armazenar os dados
acoes_df = pd.DataFrame()

# Loop para coletar os dados de fechamento usando yfinance
for acao in acoes:
    acoes_df[acao] = yf.download(acao, start='2015-01-01')['Close']

# Exibir os dados coletados


acoes_df = pd.read_csv('acoes.csv')

acoes_df['sum'] = acoes_df[['GOLL4.SA','AZUL4.SA']].mean(axis=1)


import matplotlib.pyplot as plt

acoes_df.plot(figsize=(12, 6))
plt.title('Fechamento das Ações')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.legend(acoes)
plt.show()




csv_file = 'acoes.csv'

if not os.path.exists(csv_file):
    acoes_df.to_csv(csv_file)
    print(f"Arquivo '{csv_file}' salvo com sucesso!")
else:
    print(f"O arquivo '{csv_file}' já existe. Nenhuma ação realizada.")


acoes_df.describe()






sns.histplot(acoes_df['GOLL4.SA'],bins=5)



plt.figure(figsize=(10,40))
for i in np.arange(0,len(acoes_df.columns)):
    plt.subplot(8,1,i+1)
    sns.histplot(acoes_df[acoes_df.columns[i]],kde=True)
    


