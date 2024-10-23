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
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold,KFold,RepeatedKFold,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle as pkl

"""CLASSIFICAR EMPRESAS PARA INVESTIMENTO A LONGO PRAZO"""
dataset = pd.read_excel('database//BD Completo.xlsx')
plt.style.use('Solarize_Light2')


"""-------------------------------- TRATAR VALORES FALTANTES --------------------------------"""

sns.heatmap(dataset.isnull())


dataset.isnull().sum()
dataset.columns = dataset.columns.str.strip().str.upper()

for i in dataset.columns:
    if dataset[i].isnull().sum() > 200:
        dataset = dataset.drop(f'{i}',axis=1)


dataset.isnull().sum()
sns.heatmap(dataset.isnull())



#SELECIONAR CATEGORIGOS E NUMÉRICOS
cat_attribs = dataset.select_dtypes(include=['object']).columns.tolist()
num_attribs = dataset.select_dtypes(include=['number']).columns.tolist()

#IMPUTAR MÉDIA E DROPNA- DEPOIS
dataset[num_attribs] = dataset[num_attribs].fillna(dataset[num_attribs].mean(),axis=0)
dataset.isnull().sum()

del dataset['MAJORITAR.']

dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)



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

#LIMPAR SEGMENTOS DA COLUNA 'CATEGORIA'
unique_segments, counts = np.unique(dataset['CATEGORIA'],return_counts=True)

def corrige_categoria(texto):
    categoria = ''
    if texto == 'crescimento ':
        categoria = 'crescimento'

    else:
        categoria = texto
    return categoria

dataset['CATEGORIA'] = dataset['CATEGORIA'].apply(corrige_categoria)

np.unique(dataset['CATEGORIA'],return_counts=True)


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




#DESCRIBE DATA
dataset.describe()
dataset[dataset['LPA DESCONCTADO'] > 160] 



#HISTOGRAMA
fig = plt.figure(figsize=(15,20))
ax = fig.gca()
dataset.hist(ax=ax);



#CORRELAÇÃO e DROPS
correlacao = dataset.corr(numeric_only=True)

plt.figure(figsize=(20,50))
sns.heatmap(correlacao,annot=True,cbar=False)


dataset.columns
dataset = dataset.drop(['REC. LIQUIDA','CAIXA'],axis=1)


dataset = dataset.drop(['DIVIDA BRUTA','LPA','CAIXA.1'],axis=1)
correlacao = dataset.corr(numeric_only=True)
plt.figure(figsize=(20,50))


dataset = dataset.drop(['AT. CIRCULANTE','LIQ. CORRENTE'],axis=1)
correlacao = dataset.corr(numeric_only=True)

plt.figure(figsize=(20,50))
sns.heatmap(correlacao[correlacao.abs() > 0.5], annot=True, cmap='coolwarm', cbar=False)




#ONE HOT ENCODER E VARIAVEIS DUMMIES
dataset_original = dataset.copy()
dataset_original['SITUAÇÃO'].value_counts()

cat_attribs = dataset.select_dtypes(include=['object']).columns.tolist()
num_attribs = dataset.select_dtypes(include=['number']).columns.tolist()

y = dataset['SITUAÇÃO'].values
empresas = dataset['EMPRESA']
X_cat = dataset[['SEGMENTO','CATEGORIA']]

encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(X_cat)

#TRANSFORMAÇÃO EM DATAFRAME PANDAS
X_cat = pd.DataFrame(X_cat)

dataset = dataset.drop(['SEGMENTO','CATEGORIA','SITUAÇÃO','EMPRESA'],axis=1)


#CONCATENAÇÃO DE DATAFRAMES
dataset.index
X_cat.index
dataset = pd.concat([dataset,X_cat],axis=1)


#NORMALIZAÇÃO
scaler = MinMaxScaler()
dataset.columns = dataset.columns.astype(str)
dataset_norma = scaler.fit_transform(dataset)

X = dataset_norma.copy()



#VALIDAÇÃO CRUZADA
#--------------------------------------------MODELOS--------------------------------------------
rnd_class = RandomForestClassifier(random_state=42,verbose=1)
mlp_class = MLPClassifier(hidden_layer_sizes=(175,175),verbose=1,random_state=42)
resultados_random_forest = []
resultados_random_forest_estratificado = []
resultados_rede_neural_estratificado = []

#-----------------------------------------------------------------------------------------------

#RANDOM FOREST#
rskfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=2)
scores = cross_val_score(rnd_class, X, y, cv=rskfold)
print(f"Desempenho Médio Random Forest Estratificado: {np.mean(scores)}")
resultados_random_forest_estratificado.append(scores)


rkfold = RepeatedKFold(n_splits=10,n_repeats=30,random_state=42)
scores = cross_val_score(rnd_class, X, y, cv=rkfold)
resultados_random_forest.append(scores)

print(f"Desempenho Médio Random Forest Não-Estratificado: {np.mean(scores)}")
#-----------------------------------------------------------------------------------------------
#REDE NEURAL
scores = cross_val_score(mlp_class,X,y,cv=rskfold)
print(f"Desempenho Médio Multilayer Perceptron Estratificado: {np.mean(scores)}")
resultados_rede_neural_estratificado.append(scores)


np.mean(resultados_random_forest_estratificado),np.mean(resultados_rede_neural_estratificado)




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
random_forest = RandomForestClassifier()
random_forest.fit(X_train,y_train)
previsoes = random_forest.predict(X_test)

accuracy_score(y_test,previsoes)

random_forest.classes_
cm = confusion_matrix(y_test,previsoes)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,previsoes))


#recall - identifica corretamente
#precision - classificar corretamente

#FEATURE IMPORTANCE


for nome,importancia in zip(dataset.columns,random_forest.feature_importances_):
    print(f'{nome} :  {importancia} ')




features = dataset.columns
importancias = random_forest.feature_importances_
indices = np.argsort(importancias)

plt.figure(figsize=(40,50))
plt.title('Feature Importance')
plt.barh(range(len(indices)),importancias[indices],align='center')
plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.xlabel('Importances')



#TUNING DE HIPERPARAMETROS

parametros = {'criterion':['gini','entropy'],'min_samples_split':[2,4,6,8],
            'n_estimators':[50,100,150,200]}


grid_searcher = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1,verbose=1),
                param_grid=parametros)


grid_searcher.fit(X,y)
best_params = grid_searcher.best_params_
best_score = grid_searcher.best_score_

random_forest_best = grid_searcher.best_estimator_





#TESTE DE VOTING CLASSIFIER - ENSEMBLE DE MODELOS

mlp_class = MLPClassifier(max_iter=1000)
named_estimators = [
    ("random_forest_clf", random_forest_best),
    ("mlp_clf", mlp_class),
]

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(named_estimators)  
voting_clf.fit(X,y)


previsoes = voting_clf.predict(X_test)
accuracy_score(y_test,previsoes)
#OVERFITTING

