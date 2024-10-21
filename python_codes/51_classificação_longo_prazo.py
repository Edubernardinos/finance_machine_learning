import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from pandas_datareader import data
import yfinance as yf
import os
import matplotlib.image as mpimg
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

"""CLASSIFICAR EMPRESAS PARA INVESTIMENTO A LONGO PRAZO"""
dataset = pd.read_excel('database//BD Completo.xlsx')
plt.style.use('Solarize_Light2')


"""-------------------------------- TRATAR VALORES FALTANTES --------------------------------"""

sns.heatmap(dataset.isnull())
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')
dataset.isnull().sum()
dataset.columns = dataset.columns.str.strip().str.upper()

for i in dataset.columns:
    if dataset[i].isnull().sum() > 200:
        dataset = dataset.drop(f'{i}',axis=1)


dataset.isnull().sum()

#SELECIONAR CATEGORIGOS E NUMÉRICOS
cat_attribs = dataset.select_dtypes(include=['object']).columns.tolist()
num_attribs = dataset.select_dtypes(include=['number']).columns.tolist()

#IMPUTAR MÉDIA E DROPNA- DEPOIS
dataset[num_attribs] = dataset[num_attribs].fillna(dataset[num_attribs].mean(),axis=0)
dataset.isnull().sum()
dataset = dataset.dropna()
del dataset['MAJORITAR.']


#LIMPAR SEGMENTOS DA COLUNA SEGMENTO
sns.countplot(x='SITUAÇÃO', data=dataset,palette='Set1')
unique_segments, counts = np.unique(dataset['SEGMENTO'],return_counts=True)
dataset['SEGMENTO'] = dataset['SEGMENTO'].astype(str)



def corrige_segmentos(texto):
    segmento = ''
    
    if texto == 'acessórios':
        segmento = 'acessorio'

    elif texto == 'acessorios':
        segmento = 'acessorio'

    elif texto == 'agriculltura':
        segmento = 'agricultura'

    elif texto == 'alimentos diversos':
        segmento = 'alimentos'
    
    elif texto == 'eletrodomésticos':
        segmento = 'eletrodomesticos'

    elif texto == 'equipamentos e serviços':
        segmento = 'equipamentos'

    elif texto == 'mateial rodoviario': 
        segmento = 'material rodoviario'

    elif texto == 'ser med hospit analises e diagnosticos' or texto == 'serv med hospit analises e disgnosticos' or texto == 'serv.med.hospit.analises e diagnosticos':
        segmento = 'hospitalar'

    elif texto == 'serviços de apoio e armazenamento':
        segmento = 'serviços de apoio e armazenagem'

    elif texto == 'serviços diversos s.a ctax':
        segmento = 'serviços diversos'

    elif texto == 'siderurgia':
        segmento = 'siderurgica'

    elif texto == 'soc. Credito e financiamento' or texto == 'soc credito e financiamento':
        segmento = 'credito'

    elif texto == 'tansporte aereo':
        segmento = 'transporte aereo'

    else :
        segmento = texto

    return segmento
dataset['SEGMENTO'] = dataset['SEGMENTO'].apply(corrige_segmentos)

#LIMPAR SEGMENTOS DA COLUNA CATEGORIA
unique_segments, counts = np.unique(dataset['CATEGORIA'],return_counts=True)

def corrige_categoria(texto):
    categoria = ''
    if texto == 'crescimento ':
        categoria = 'crescimento'

    else:
        categoria = texto
    return categoria

dataset['CATEGORIA'] = dataset['CATEGORIA'].apply(corrige_categoria)


plt.figure(figsize=(10,10))
sns.countplot(x='CATEGORIA', data=dataset,palette='Set1')
plt.xticks()



#guardar para depois pipeline
"""class PreprocessPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, num_attribs, cat_attribs, drop_na=False):
        self.num_attribs = num_attribs
        self.cat_attribs = cat_attribs
        self.drop_na = drop_na
        
        #  numérico
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
        ])
        
        # categórico
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputação com a moda
            ('cat_encoder', OneHotEncoder(sparse_output=False)),  
        ])

    def fit(self, dataset, y=None):
        self.num_pipeline.fit(dataset[self.num_attribs])
        
        if not self.drop_na:
            self.cat_pipeline.fit(dataset[self.cat_attribs])

        return self

    def transform(self, dataset, y=None):
        # Transformar as colunas numéricas
        num_data = self.num_pipeline.transform(dataset[self.num_attribs])
        
        if self.drop_na:
            dataset = dataset.dropna(subset=self.cat_attribs)
            cat_data = pd.DataFrame()  # Placeholder para manter a estrutura do código

        else:
            cat_data = self.cat_pipeline.transform(dataset[self.cat_attribs])
        

        return pd.concat([pd.DataFrame(num_data, index=dataset.index), 
                          pd.DataFrame(cat_data, index=dataset.index)], axis=1)
    
"""


