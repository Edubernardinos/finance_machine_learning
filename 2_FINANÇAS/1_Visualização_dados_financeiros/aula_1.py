import requests
import pandas as pd

# Sua chave da API
api_key = 'IUHRZDKZFQGHVCFD'
ticker = 'GOLL4.SA'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&datatype=csv'

# Fazendo a requisição para a API
response = requests.get(url)

# Verificando se a requisição foi bem-sucedida
if response.status_code == 200:
    # Salvar os dados em um arquivo CSV
    with open(f'{ticker}_data.csv', 'wb') as file:
        file.write(response.content)
    print(f'Dados salvos em {ticker}_data.csv')
else:
    print(f'Erro ao buscar dados: {response.status_code}, {response.text}')

# Carregar os dados do CSV
gol_df = pd.read_csv(f'{ticker}_data.csv')


