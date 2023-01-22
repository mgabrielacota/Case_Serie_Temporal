# Databricks notebook source
# MAGIC %md
# MAGIC # Problema: Previsão de série temporal 
# MAGIC Projete a demanda para cada produto nas lojas para os próximos 30, 60 e 90 dias!
# MAGIC 
# MAGIC  
# MAGIC  The first step when initiating the demand forecasting project is to provide the client with meaningful insights. The process includes the following steps:
# MAGIC 
# MAGIC Reunir os dados disponíveis
# MAGIC Olhar a estrutura dos dados, acurácia e consistência
# MAGIC Rodar testes e pilotos 
# MAGIC Avaliar o resumo estatistico

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

#modelo
import xgboost as xgb
#metricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#tunning de hiperparametros
from sklearn.model_selection import RandomizedSearchCV

# COMMAND ----------

df_canais = pd.read_csv("dados/Canais.csv", delimiter=";")
df_lojas = pd.read_csv("dados/Lojas.csv", delimiter=";")
df_prod = pd.read_csv("dados/Produtos.csv", delimiter=";")
df_unidades = pd.read_csv("dados/Unidades Negócios.csv", delimiter=";")
df_vendas = pd.read_csv("dados/Vendas.csv", delimiter=";")

# COMMAND ----------

df_canais #por onde a venda foi feita; ex: ecomerce ou loja fisica

# COMMAND ----------

df_lojas 

# COMMAND ----------

df_prod

# COMMAND ----------

df_unidades

# COMMAND ----------

df_vendas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Juntar dataframes em um só, usando chave-primária
# MAGIC Antes de mais nada: quais são as quantidades que terei disponiveis no futuro para fazer uma previsao sobre venda?
# MAGIC * produto
# MAGIC * data (e demais variaveis relacionadas - feature engeneering)
# MAGIC * lag (feature engereering)
# MAGIC 
# MAGIC Dado isso, utilizaremos *apenas* os dados que estao nos dataframes de vendas e produtos.

# COMMAND ----------

df = df_vendas.merge(df_prod, left_on="id_produto", right_on="produto", how="left")

# COMMAND ----------

df.groupby(["id_produto"]).size().reset_index()

# COMMAND ----------

df

# COMMAND ----------

df.info() #vamos analizar os nossos dados

# COMMAND ----------

# MAGIC %md
# MAGIC #### Como esse é um problema de *demanda* por produto ao longo do tempo, vamos manter apenas as colunas de:
# MAGIC * id_data
# MAGIC * produto_nome (poderia ser id_produto tambem ou produto, mas a coluna produto_nome é mais legivel)
# MAGIC * qtde_venda

# COMMAND ----------

columns_to_keep = ['id_data','qtde_venda','produto_nome']
columns_to_drop = [i for i in df.columns if i not in columns_to_keep]
    
print(columns_to_drop)

# COMMAND ----------

df.drop(columns=columns_to_drop, axis=1, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabela em ordem cronologica
# MAGIC Para facilitar a vizualização

# COMMAND ----------

df.sort_values("id_data", inplace=True)
df.reset_index(inplace=True)

# colocar datas em unidades de dias à partir do primeiro dia

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizar dados

# COMMAND ----------

#verificar se existe mais de um par produto-data
df.groupby(['produto_nome', 'id_data']).size().reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC Existe mais de uma linha para alguns pares "produto-data". Como quero prever a demanda total de cada produto por data, entao terei que agrupar essas linhas repetidas, para fazer a tabela mostrar a qtde_vendas total por produto por dia!

# COMMAND ----------

df["qtde_venda"] = df["qtde_venda"].apply(lambda x: x.replace(',', '.')).astype('float')

# COMMAND ----------

dados = df.groupby(['produto_nome', 'id_data'])['qtde_venda'].sum().unstack(fill_value=0).stack().reset_index()
#dados = df.groupby(['produto_nome', 'id_data'])['qtde_venda'].sum().reset_index()

dados.rename(columns={0:"venda_total"}, inplace=True)
dados.reset_index() #deletou dias com quantidade de venda 0 !!!

# COMMAND ----------

# MAGIC %md
# MAGIC Assim, criamos a **target**: venda_total por produto por data, onde produto e data sao os valores que iremos passar ao modelo

# COMMAND ----------

dados.info()

# COMMAND ----------

#5432
dados_plot = dados[dados['produto_nome'] == "Produto 5432"][["id_data", "venda_total"]]

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos plotar o total de vendas de um produto aleatorio ao longo de todo o tempo

# COMMAND ----------

dados_plot.plot(x='id_data', y='venda_total', style='.', figsize=(10, 8))

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos plotar para o periodo de um mes: março de 2018

# COMMAND ----------

dados_plot.loc[(dados_plot.id_data >= "2018-03-01") & (dados_plot.id_data < "2018-04-01")]\
          .plot(x='id_data', y='venda_total', style='.', figsize=(10, 8))

# COMMAND ----------

# MAGIC %md
# MAGIC Esse produto tem muitos valores nulos, esses valores provavelmente representam a nao venda daquele produto naquele dia. Isso pode atrapalhar o modelo

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engeneering
# MAGIC Primeiro, vamos criar features afim de conseguir obter um modelo melhor - com melhor acurácia. Posteriormente, selecionaremos quais dessas features sao de fato relevantes para a analise e devem permanecer. Nossas features devem ser capazes de detectar 
# MAGIC * **padrões sazonais**: padrões regulares em curtos periodos de tempo
# MAGIC * **padrões cíclicos**: padrões regulares em longos periodos de tempo

# COMMAND ----------

dados.groupby("produto_nome").size().reset_index().sort_values(0)

# COMMAND ----------

dados[dados['produto_nome'] == 'Produto 7273'] \
     .plot(x='id_data', y='venda_total', style='.',
          figsize=(15, 5),
          title='Produto 7273')

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos ver se conseguimos tirar algum tipo de tendencia

# COMMAND ----------

def media_movel_futuro(df, janelas):
    
    'calcula media movel futura'
    df = df.copy()
    df = df.iloc[::-1] #inverter tabela
    
    for window in janelas:
        #faz o media movel do "passado"
        df['venda_roll_mean_' + str(window)] = df.groupby(["produto_nome"])['venda_total']\
                                                 .transform(lambda x: x.shift(1)\
                                                 .rolling(window=window, min_periods=window)\
                                                 .mean().reset_index(0,drop=True))
        
    
    return df.iloc[::-1] #inverte de volta

# COMMAND ----------

dados_7273 = dados[dados['produto_nome'] == 'Produto 7273']
dados_7273 = media_movel_futuro(dados_7273,[7,15])

# COMMAND ----------

dados_7273 

# COMMAND ----------

fig, ax = plt.subplots(figsize=(15, 5))

dados[dados['produto_nome'] == 'Produto 7273'] \
     .plot(ax=ax, x='id_data', y='venda_total', style='.',
          title='Produto 7273', label='dado originais')

dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_7', style='-', color="red", label='soma movel: 7 dias'
          )


ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos ver que os dados originais nao tem padroes detectaveis. No entando, quando fazemos uma soma ou media movel, tendencias comecam a aparecer. Por esse motivo, para o modelo performar bem, utilizaremos **media ou soma movel de vendas por dia por produto como target**
# MAGIC 
# MAGIC 
# MAGIC ## $y \equiv \text{soma movel em x dias de vendas, por dia, por produto}$

# COMMAND ----------

dados_7273 = media_movel_futuro(dados_7273,[21,30])

# COMMAND ----------

fig, ax = plt.subplots(figsize=(15, 5))


dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_7', style='-', color="red", label='soma movel: 7 dias'
          )

dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_15', style='-', color="black", label='soma movel: 15 dias'
          )

dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_21', style='-', color="blue", label='soma movel: 21 dias'
          )

dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_30', style='-', color="orange", label='soma movel: 30 dias'
          )



ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 30 dias suaviza demais os eventos individuais, algo entre 15 e  21 arece uma boa opcao

# COMMAND ----------

fig, ax = plt.subplots(figsize=(15, 5))


dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_15', style='-', color="red", label='soma movel: 15 dias'
          )

dados_7273\
     .plot(ax=ax, x='id_data', y='venda_roll_mean_21', style='-', color="black", label='soma movel: 21 dias'
          )




ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos escolher 15 por praticidade: ja que o menor intervalo de tempo que iremos prever sera de 30 dias. Assim ficamos com
# MAGIC 
# MAGIC ## $y \equiv \text{soma movel dos proximos 15 dias de vendas, por dia, por produto}$

# COMMAND ----------

### Target
def soma_movel_futuro(df, janelas):
    
    'calcula media movel futura'
    df = df.copy()
    #inverter tabela (dados recentes no inicio e dados antigos no final)
    df = df.iloc[::-1] 
    
    for window in janelas:
        #faz o media movel do "passado"
        df['target_soma_movel_' + str(window)] = df.groupby(["produto_nome"])['venda_total']\
                                                 .transform(lambda x: x.shift(1)\
                                                 .rolling(window=window, min_periods=window)\
                                                 .sum().reset_index(0,drop=True))
        
    
    return df.iloc[::-1] #inverte de volta

# COMMAND ----------

dados = soma_movel_futuro(dados, [15])

# COMMAND ----------

#checando se esta tudo certo
dados[dados['produto_nome'] == 'Produto 9999'].tail(20)

# COMMAND ----------

dados[dados['produto_nome'] == 'Produto 9999'].tail(35)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features de tempo
# MAGIC Queremos features que detectem padroes cíclicos

# COMMAND ----------

#colocar datas no formato de datetime do Pandas para facilitar operacoes
dados['id_data'] = pd.to_datetime(dados.id_data, format='%Y-%m-%d')

# COMMAND ----------

def criar_features_de_tempo(dados=dados):
    """
    Cria features de series temporais.
    """
    dados = dados.copy()
    #df['hour'] = df.index.hour
    dados["diadasemana"]  = dados.id_data.dt.dayofweek
    dados["diadomês"]     = dados.id_data.dt.day
    dados["diadoano"]     = dados.id_data.dt.dayofyear
    dados["mês"]          = dados.id_data.dt.month
    dados["semanadoano"]  = dados.id_data.dt.isocalendar().week.astype("int64")
    
    # 0: Verao - 1: Outono - 2: Inverno - 3: Primavera
    dados["estação"] = np.where(dados.mês.isin([12,1,2]), 0, 1)
    dados["estação"] = np.where(dados.mês.isin([6,7,8]), 2, dados["estação"])
    dados["estação"] = np.where(dados.mês.isin([9, 10, 11]), 3, dados["estação"])

    return dados

# COMMAND ----------

dados1 = criar_features_de_tempo()
dados1.reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduzir Lag Features
# MAGIC Lag features são valores em tempos anteriores que são consideradas úteis por assumirmos que o que aconteceu no passado pode influenciar ou reter algum tipo de informação intrínseca sobre o futuro.
# MAGIC 
# MAGIC Lembrando que capturar a tendencia e sazonalidade é o nosso objetivo principal

# COMMAND ----------

def lag_features(df, lags):
    for lag in lags:
        df['lag_vendas_' + str(lag)] = df.groupby(["produto_nome"])['target_soma_movel_15']\
                                         .transform(lambda x: x.shift(lag))
    return df

# COMMAND ----------

#o spam de tempo que queremos calcular é de 30, 60, 90 dias, 
#o banco de dados vai ate 730 dias, entao
#vamos adicionar multiplos de 30, 60, 90, ate 90 dias 
#para nao perdermos muitos dados no conjunto de teste

lags = np.arange(7, 95, 7)

# COMMAND ----------

dados1 = lag_features(dados1, lags)
dados1.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features de média movel
# MAGIC Vamos calcular a média movel. Esse tipo de é bom para remover as pequenas variações entre time steps - cria uma versao suavizada do dataset

# COMMAND ----------

def features_media_mov(df, janelas):
    for window in janelas:
        df['venda_media_movel_' + str(window)] = df.groupby(["produto_nome"])['venda_total']\
                                                 .transform(lambda x: x.shift(1)\
                                                 .rolling(window=window, min_periods=window)\
                                                 .mean())
    return df

# COMMAND ----------

janelas = [7, 15, 35, 65, 95, 120]

# COMMAND ----------

# MAGIC %md
# MAGIC Pra evitar que a previsao fique ruim, vou considerar janelas maiores que 90 dias, de forma que quando fizer a previsao de 90 dias, eu consiga considerar parte dos valores do conjunto de dados para fazer a media (e nao fazer a media apenas nos valores que o modelo prever)

# COMMAND ----------

dados1= features_media_mov(dados1, janelas)

# COMMAND ----------

dados1.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Training, validacao e Test Split
# MAGIC A maior janela é de 120 dias, entao a janela de treino começara no dia 180 
# MAGIC Faremos validacao cruzada de uma forma diferente com a qual estamos separando conjuntos de treino e teste

# COMMAND ----------

train = dados1[dados1['id_data'].between("2018-05-01","2019-07-01")] #~10^6 linhas!
cv    = dados1[dados1['id_data'].between("2019-07-02","2019-10-01")]
test  = dados1[dados1['id_data'].between("2019-10-02","2019-12-31")]

# COMMAND ----------

#train = train.sample(5*10**5, random_state=123)
train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analise de Outliers

# COMMAND ----------

train.groupby(["produto_nome"]).agg({"target_soma_movel_15": ["count","sum", "mean", "median", "std", "min", "max"]})

# COMMAND ----------

train["target_soma_movel_15"].plot(kind='hist', bins=50)

# COMMAND ----------

print("Numero max de vendas de todos os produtos por dia:", train["target_soma_movel_15"].max(),"\n",\
      "Numero min de vendas de todos os produtos por dia:", train["target_soma_movel_15"].min())

# COMMAND ----------

print(train.query('target_soma_movel_15 < 0').count()['target_soma_movel_15'],\
      train.query('target_soma_movel_15 > 200').count()['target_soma_movel_15'])

# COMMAND ----------

train.query('target_soma_movel_15 < 0') \
    .plot(x='id_data', y='target_soma_movel_15', style='.',
          figsize=(15, 5),
          title='Outliers')

# COMMAND ----------

# MAGIC %md
# MAGIC Remover esses dados estranho do periodo de nov 2018

# COMMAND ----------

train = train.query('target_soma_movel_15 > -4').copy() #remover outlier

#plotar nova distribuicao
train.query('target_soma_movel_15 < 0') \
    .plot(x='id_data', y='target_soma_movel_15', style='.',
          figsize=(15, 5),
          title='Outliers')

# COMMAND ----------

train.query('target_soma_movel_15 > 200') \
    .plot(x='id_data', y='target_soma_movel_15', style='.',
          figsize=(15, 5),
          title='Outliers')

# COMMAND ----------

# MAGIC %md
# MAGIC Nao parece haver nada de visilvelmente anormaal aqui. Vamos manter todos os dados

# COMMAND ----------

# MAGIC %md
# MAGIC # Categorical Encoding
# MAGIC Vamos transformar as colunas de *categorias* (colunas que mesmo se representadas por numeros, os numeros nao tem valor quantitativo):
# MAGIC * produto_nome
# MAGIC * diadasemana
# MAGIC * diadomês 
# MAGIC * diadoano 
# MAGIC * mês 
# MAGIC * semanadoano
# MAGIC * estação
# MAGIC 
# MAGIC em colunas binarias para que o modelo funcione.

# COMMAND ----------

#como produto_nome tem milhares de categorias, antes de aplicar a transformacao vamos
#ver quanto de nosso dataframe ocupa
BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(train)

# COMMAND ----------

# MAGIC %md
# MAGIC o dataframe transformado vai ocupar muito mais do que o dataframe original ocupa de RAM. Portanto, vamos tentar aliviar esse efeito usando **TARGET ENCODING**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Encoder
# MAGIC A ideia principal aqui é fazer um encode das categorias por substitui-las por uma *medida do efeito que elas terao na target* venda_total por produto por dia

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Vamos substituir a categoria pelo valor da média do target para aquela categoria e, para evitar overfitting, vamos fazer **additive smoothing** 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additive Smoothing
# MAGIC "Suavizar” a media ao incluir as vendas de *todos os produtos*
# MAGIC 
# MAGIC # $\mu = \frac{n\times\bar{x}\,+\,m\times w}{n+m}$
# MAGIC onde
# MAGIC * $\mu$ é a média que vai substituir os valores categoricos
# MAGIC * $n$ é o numero de valores
# MAGIC * $\bar{x}$ é a media estimada para a categoria
# MAGIC * $m$ é o peso que vamos associar à media total. Quanto maior m maior o peso da media total
# MAGIC * $w$ é a media total

# COMMAND ----------

def calc_media_suave(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)

# COMMAND ----------

train_encod = train.copy()

# o encoding do test e cv deve ser basado no target encoding para o conj. de treino 
cv_encod    = cv.copy()
test_encod  = test.copy()

# COMMAND ----------

features = ["produto_nome",
"diadasemana",
"diadomês",
"diadoano",
"mês",
"semanadoano",
"estação"] 

novas_features = ["Nproduto_nome",
"Ndiadasemana",
"Ndiadomês",
"Ndiadoano",
"Nmês",
"Nsemanadoano",
"Nestação"] 

# COMMAND ----------

m = 100
for i in range(len(features)):

    train_encod[ novas_features[i] ]=calc_media_suave(train_encod, by=features[i], on='target_soma_movel_15', m=m)
#m=100 chute inicial

# COMMAND ----------

print_memory_usage_of_data_frame(train_encod)

# COMMAND ----------

train_encod.info()

# COMMAND ----------

# MAGIC %md
# MAGIC criar tabelas de correspondencia entre valor e categoria e aplicalas nos conj de teste e cv

# COMMAND ----------

corresp_prod    = train_encod.groupby(['produto_nome', 'Nproduto_nome'])\
                             .size().reset_index()[['produto_nome', 'Nproduto_nome']]

corresp_semana  = train_encod.groupby(['diadasemana', 'Ndiadasemana'])\
                             .size().reset_index()[['diadasemana', 'Ndiadasemana']]

corresp_dmes     = train_encod.groupby(['diadomês', 'Ndiadomês'])\
                             .size().reset_index()[['diadomês', 'Ndiadomês']]
 
corresp_ano     = train_encod.groupby(['diadoano', 'Ndiadoano'])\
                             .size().reset_index()[['diadoano', 'Ndiadoano']]

corresp_mes     = train_encod.groupby(['mês', 'Nmês'])\
                             .size().reset_index()[['mês', 'Nmês']]

corresp_sem_ano = train_encod.groupby(['semanadoano', 'Nsemanadoano'])\
                             .size().reset_index()[['semanadoano', 'Nsemanadoano']]

corresp_estacao =train_encod.groupby(['estação', 'Nestação'])\
                             .size().reset_index()[['estação', 'Nestação']]

# COMMAND ----------

corresp = [corresp_prod, corresp_semana, corresp_dmes, corresp_ano, corresp_mes, corresp_sem_ano, corresp_estacao]

# COMMAND ----------

for i in range(len(features)):
    test_encod = test_encod.merge(corresp[i], on=features[i], how="left")
    cv_encod   = cv_encod.merge(corresp[i], on=features[i], how="left")

# COMMAND ----------

# remover features categoricas, deixando apenas os seus encoded values
for i in range(len(features)):
    test_encod.drop(features[i], axis=1, inplace=True)
    cv_encod.drop(features[i], axis=1, inplace=True)
    train_encod.drop(features[i], axis=1, inplace=True)

# COMMAND ----------

test_encod.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Criando modelo: XGBReg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinando com validacao cruzada

# COMMAND ----------

FEATURES = list(dados2.columns)
remover = ['id_data',
 'venda_total',
 'target_soma_movel_15']

for i in remover:
    FEATURES.remove(i)
    
TARGET = ['target_soma_movel_15']

# COMMAND ----------

tss = TimeSeriesSplit(n_splits=5)
#df = df.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train1 = dados2.iloc[train_idx]
    val = dados2.iloc[val_idx]

    X_train = train1[FEATURES]
    Y_train = train1[TARGET]

    X_val = val[FEATURES]
    Y_val = val[TARGET]

    reg = xgb.XGBRegressor(booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           #bjective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            verbose=100)

    Y_pred = reg.predict(X_val)
    preds.append(Y_pred)
    score = np.sqrt(mean_squared_error(Y_val, Y_pred))
    scores.append(score)

# COMMAND ----------

scores

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Tunning de Hiperparametros

# COMMAND ----------

z

# COMMAND ----------

X_train = train_encod.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_train = train_encod["target_soma_movel_15"]

X_val = cv_encod.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_val = cv_encod["target_soma_movel_15"]

X_test = test3.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_test = test3["target_soma_movel_15"]

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

# COMMAND ----------

# MAGIC %%time
# MAGIC regrXGB = xgb.XGBRegressor(booster='gbtree',    
# MAGIC                            n_estimators=1000,
# MAGIC                            early_stopping_rounds=50,
# MAGIC                            objective='reg:linear',
# MAGIC                            max_depth=3,
# MAGIC                            learning_rate=0.008)
# MAGIC 
# MAGIC 
# MAGIC regrXGB.fit(X_train, Y_train,
# MAGIC         eval_set=[(X_train, Y_train), (X_val, Y_val)],
# MAGIC         verbose=100)

# COMMAND ----------

def smape(previsao, dado):
    return 100/len(dado) * np.sum(np.abs(previsao - dado) / (np.abs(dado) + np.abs(previsao)))

# COMMAND ----------

def print_metrics(X, Y):
    print( "SMAPE:", smape(regrXGB.predict(X), Y) )
    print("\tMean absolute error (MAE):", mean_absolute_error(Y, regrXGB.predict(X)))
    print("\tRoot Mean squared error (RMSE):",  np.sqrt(mean_squared_error(Y, regrXGB.predict(X))))
    print("\tR2 score:", r2_score(Y, regrXGB.predict(X)))
    
    return

# COMMAND ----------

#train metrics
print_metrics(X_train, Y_train)

# COMMAND ----------

#test metrics
print_metrics(X_val, Y_val)

# COMMAND ----------

#test metrics
print_metrics(X_test, Y_test)

# COMMAND ----------

Y_train.describe().reset_index() #comparar metricas com as estatisticas do dado

# COMMAND ----------

# MAGIC %md
# MAGIC O R2 parece ótimo para os conjuntos de validacao, treino e teste. MAE é menos do que o valor médio da target. No entanto, SMAPE parece bem problematico. Vamos plotar o fit em cima dos dados pra ajudar a entender se o modelo esta bom ou nao.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotar Resultado

# COMMAND ----------

previsao_val = regrXGB.predict(X_val)
previsao_train = regrXGB.predict(X_train)
#previsao_test = regrXGB.predict(X_test)

# COMMAND ----------

#test4 = test.dropna()

# COMMAND ----------

cv['previsao'] = previsao_val
train['previsao'] = previsao_train
#test4['previsao'] = previsao_test

# COMMAND ----------

cv_7273 = cv[cv['produto_nome'] == 'Produto 7273']
train_7273 = train[train['produto_nome'] == 'Produto 7273']
#test_7273 = test4[test4['produto_nome'] == 'Produto 7273']

# COMMAND ----------

def plot_resultados(df):
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(df['id_data'], df['target_soma_movel_15'], color="black", label='resultado verdadeiro '
              )
    ax.plot(df['id_data'], df['previsao'], color="red", label='previsao teste'
              )


    ax.legend()
    plt.show()

    return

# COMMAND ----------

#treino
plot_resultados(train_7273)

# COMMAND ----------

#validacao
plot_resultados(cv_7273)

# COMMAND ----------

# MAGIC %md
# MAGIC Olhando para o plot, podemos entender porque o SMAPE estava ruim e o R2 bom. Os modelos estao com formas bem quadradas e com muita dificuldade de generalizacao! Provavelmente o modelo esta fitando um monte de zeros
# MAGIC 
# MAGIC Na etapa de analise do banco de dados, nos ja tinhamos visto que temos poucos valores de venda por produto (ate 730 pontos) e alem disso, muitos desses produtos tem muitos valores nulos!
# MAGIC 
# MAGIC Isso indica duas possibilidades para esse overfitting: **curse of dimensionality** e **sparse data**
# MAGIC 
# MAGIC Antes de atacar esses dois pontos, vamos dar uma olhada na importancia das features para termos mais insights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importância da feature

# COMMAND ----------

feature_importance= pd.DataFrame(data   = regrXGB.feature_importances_,
                 index  = regrXGB.feature_names_in_,
                 columns= ['importance'])
feature_importance.sort_values('importance', ascending=False).head(25)

# COMMAND ----------

plt.figure(figsize=(10, 5))
fi = feature_importance.sort_values('importance', ascending=False).head(10)
fi.plot(kind='barh', title='Feature Importance')
plt.legend(loc='lower right')
plt.show()

# COMMAND ----------

rs = np.random.RandomState(0)
df = pd.DataFrame(X_train[['Nproduto_nome','venda_media_movel_95','venda_media_movel_120']])
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

# MAGIC %md
# MAGIC Ate agora, fitamos o modelo para todos os produtos ao mesmo tempo e vimos que nao conseguimos capturar as tendencias da serie temporal. Sabemos que temos por volta de 15000 produtos, muitos deles, com muitos zeros. Pode nao ser o ideal, mas reduzir o numero de produtos para, por exemplo os 100 mais vendidos e concatenar os demais, pode melhorar os resultados do modelo..  
# MAGIC 
# MAGIC Mas para ver se essa hipotese faz sentido, vamos primeiro ver se o modelo XGBoost é capaz de modelar bem quando **fitado para apenas 1 produto**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fitar modelo em um produto

# COMMAND ----------

a = float(corresp_prod[corresp_prod["produto_nome"] == "Produto 7273"]['Nproduto_nome'])

# COMMAND ----------

#fitar em um dado

train1_7273 = train_encod[train_encod['Nproduto_nome'] == a]
val1_7273 = cv_encod[cv_encod['Nproduto_nome'] == a]


X_train_7273 = train1_7273.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_train_7273 = train1_7273['target_soma_movel_15']

X_val_7273 = val1_7273.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_val_7273 = val1_7273['target_soma_movel_15']


# COMMAND ----------

# MAGIC %%time
# MAGIC regrXGB = xgb.XGBRegressor(booster='gbtree',    
# MAGIC                            n_estimators=5000,
# MAGIC                            early_stopping_rounds=50,
# MAGIC                            objective='reg:linear',
# MAGIC                            max_depth=2,
# MAGIC                            reg_lambda= 10, #acrescentar um L2 alto para ver se conseguimos evitar overvitting
# MAGIC                            learning_rate=0.008)

# COMMAND ----------

regrXGB.fit(X_train_7273, Y_train_7273,
        eval_set=[(X_train_7273, Y_train_7273), (X_val_7273, Y_val_7273)],
        verbose=100)


# COMMAND ----------

#validation
print_metrics(X_val_7273, Y_val_7273)

# COMMAND ----------

#Train
print_metrics(X_train_7273, Y_train_7273)

# COMMAND ----------

# MAGIC %md
# MAGIC **Overfitting peranece** mesmo com parametro de Regularizacao L2 alto! No entanto, o SMAPE melhorou consideravelmente. Vamos ver o que aconteceu com um plot

# COMMAND ----------


val1_7273['previsao'] = regrXGB.predict(X_val_7273)
train1_7273['previsao'] = regrXGB.predict(X_train_7273)


# COMMAND ----------

#validacao
plot_resultados(val1_7273)

# COMMAND ----------

#treino
plot_resultados(train1_7273)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que aquele padrao "quadrado" do modelo desapareceu. Esses plots junto com a metrica me dizem que provavelmente a grande quantidade de produtos com poucos pontos é um problema. No entanto, a permanencia do overfitting mesmo com regularizacao, parece apontar para o problema de **curse of dimensionaality**. Pra atacar a **curse of dimensionality** eu penso em duas opcoes imediatas:
# MAGIC 
# MAGIC * 1. Testar o modelo xgboost com menos dimensoes (remover features menos importantes e ver como o modelo se comporta)
# MAGIC * 2. Caso 1 nao funcione, tentar um modelo estatistico para a serie temporal

# COMMAND ----------

feature_importance= pd.DataFrame(data   = regrXGB.feature_importances_,
                 index  = regrXGB.feature_names_in_,
                 columns= ['importance'])

# COMMAND ----------

plt.figure(figsize=(10, 5))
fi = feature_importance.sort_values('importance', ascending=False).head(6)
fi.plot(kind='barh', title='Feature Importance')
plt.legend(loc='lower right')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reducao de dimensao
# MAGIC Fazer um PCA provavelmente seria uma solucao mais elegante, que podemos implementar no futuro, mas por hora, vamos somente manter as 5 features mais importantes

# COMMAND ----------

#testar modelo com as 5 features mais importantes
X2_train_7273 = X_train_7273[["lag_vendas_7", "lag_vendas_21", "venda_media_movel_35",\
                            "venda_media_movel_65", "Nproduto_nome"]]

X2_val_7273 = X_val_7273[["lag_vendas_7", "lag_vendas_21", "venda_media_movel_35",\
                            "venda_media_movel_65", "Nproduto_nome"]]

# COMMAND ----------

regrXGB.fit(X2_train_7273, Y_train_7273,
        eval_set=[(X2_train_7273, Y_train_7273), (X2_val_7273, Y_val_7273)],
        verbose=100)

# COMMAND ----------

#validacao
print_metrics(X2_val_7273, Y_val_7273)

# COMMAND ----------

#train
print_metrics(X2_train_7273, Y_train_7273)

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE e MAE mostram que manter apenas as 5 features mais importantes melhora consideravelment melhora no problema de overfitting! Os valores de SMAPE tambem melhoraram e o R2 do conjunto de validacao tambem mostra uma expressiva **melhora na generalizacao**!!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plotar resultados

# COMMAND ----------

train2_7273 = train_encod[train_encod['Nproduto_nome'] == a]
val2_7273 = cv_encod[cv_encod['Nproduto_nome'] == a]

val2_7273['previsao'] = regrXGB.predict(X2_val_7273)
train2_7273['previsao'] = regrXGB.predict(X2_train_7273)
#test4['previsao'] = previsao_test

# COMMAND ----------

#validacao
plot_resultados(val2_7273)

# COMMAND ----------

#comparar com e sem a reducao de dimensao
#validacao
fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(val1_7273['id_data'], val1_7273['target_soma_movel_15'], color="black", label='resultado verdadeiro '
          )
ax.plot(val1_7273['id_data'], val1_7273['previsao'], color="red", label='previsao todas as features'
          )
ax.plot(val1_7273['id_data'], val2_7273['previsao'], color="blue", label='previsao 5 features'
          )


ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC com essas 5 features, as amplitudes da previsao se aproximam mais da dos dados

# COMMAND ----------

#treino
plot_resultados(train2_7273)

# COMMAND ----------

# MAGIC %md
# MAGIC essao sao os resultados ao fitar *um produto* agora vamos ver as melhoras que obtemos atravez da reducao de dimensao com todos os produtos ao mesmo tempo!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo com dimensoes reduzidas!

# COMMAND ----------

#modificar X_train e X_val

X_train_reduzido = X_train[["lag_vendas_7", "lag_vendas_21", "venda_media_movel_35",\
                            "venda_media_movel_65", "Nproduto_nome"]]


X_val_reduzido   =  X_val[["lag_vendas_7", "lag_vendas_21", "venda_media_movel_35",\
                            "venda_media_movel_65", "Nproduto_nome"]]

# COMMAND ----------

X_train_reduzido.info()

# COMMAND ----------

regrXGB = xgb.XGBRegressor(booster='gbtree',    
                           n_estimators=2000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=2,    #arvores mais rasas para evitar overvitting
                           reg_lambda= 10, #acrescentar um L2 alto para ver se conseguimos evitar overvitting
                           learning_rate=0.008)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC regrXGB.fit(X_train_reduzido, Y_train, 
# MAGIC         eval_set=[(X_train_reduzido, Y_train), (X_val_reduzido, Y_val)],
# MAGIC         verbose=100)

# COMMAND ----------

regrXGB.predict(X_train_reduzido)

# COMMAND ----------

5864403, 5864403

# COMMAND ----------

cv_reduzido    = cv.drop('previsao', axis=1) 
train_reduzido = train.drop('previsao', axis=1)

cv_reduzido['previsao']    = regrXGB.predict(X_val_reduzido)
train_reduzido['previsao'] = regrXGB.predict(X_train_reduzido)

# COMMAND ----------

cv_reduzido.info()

# COMMAND ----------

#treino
print_metrics(X_train_reduzido, Y_train)

# COMMAND ----------

#validacao
print_metrics(X_val_reduzido, Y_val)

# COMMAND ----------

train3_7273 = train_reduzido[train_reduzido['produto_nome'] == 'Produto 7273']
val3_7273 = cv_reduzido[cv_reduzido['produto_nome'] == 'Produto 7273']

# COMMAND ----------

#treino
plot_resultados(train3_7273)

# COMMAND ----------

#Validacao
plot_resultados(val3_7273)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparse data
# MAGIC Parece que as melhoras que obtivemos ao reduzir as dimensoes foram dominadas por outro efeito. Vamo ver se, reduzir o numero de produtos e lidar com os com muitos produtos de valores zero ajuda.
# MAGIC 
# MAGIC Here is what you could try doing:
# MAGIC 
# MAGIC 
# MAGIC Normalize your daily data. Subtract the daily series by the "daily mean" and divide by the "daily standard deviation". Run a clustering algorithm (perhaps K-Means) on your daily timeseries. Use the elbow to identify the best number of clusters.
# MAGIC 
# MAGIC Plot the centroids of your clusters. If you are lucky - you may be able to see distinct shapes that your forecast curve takes.
# MAGIC 
# MAGIC Use the Cluster Number to label each daily time series. Then use a classification model to predict the correct cluster. The features for the classification model could be "day of week", "is_holiday", "expected_average_temperature_for_the_day" etc etc.
# MAGIC 
# MAGIC Check if your classification model does a reasonably good job. If it does you are probably in luck. Your classification model assigns probabilities for each cluster. Combine the cluster centroids weighted by the predicted probabilities - to arrive at a predicted curve.
# MAGIC 
# MAGIC The predicted curve from the previous step is probably normalized - since the data-prep step (Step 3) normalized the data. You now have the task to rescale back to original data. From Step 2 - if you were able to construct a model that does a reasonable job at predicting the "mean" and "variance" for the target day - Then you could do something as simple as: Final_Curve = Normalized_Curve * sqrt(Forecasted_Variance) + Forecasted_Mean

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos contar o numero vendas de cada produto ao longo de todo o tempo

# COMMAND ----------

df1 = df_vendas.merge(df_prod, left_on="id_produto", right_on="produto", how="left")

# COMMAND ----------

df1.info()

# COMMAND ----------

dados_vazios = df1.groupby(['produto_nome', 'id_data'])['qtde_venda']\
                 .sum().reset_index()

# COMMAND ----------

dados_vazios["qtde_venda"] = df1["qtde_venda"].apply(lambda x: x.replace(',', '.')).astype('float')
dados_vazios.rename(columns={0:"venda_total"}, inplace=True)

# COMMAND ----------

freq_prod = dados_vazios.produto_nome.value_counts().reset_index()

# COMMAND ----------

freq_prod['frequencia'] = freq_prod['produto_nome']
freq_prod['produto_nome'] = freq_prod['index']
freq_prod.drop(['index'], axis=1, inplace=True)
freq_prod

# COMMAND ----------

dados_vazios.rename(columns={0:"venda_total"}, inplace=True)

# COMMAND ----------

freq_prod[freq_prod['frequencia'] >= 365].count()

# COMMAND ----------

150/730*100

# COMMAND ----------

freq_prod[freq_prod['frequencia'] <= 150].count()

# COMMAND ----------

10316/13734*100

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos ver que mais de 75% dos produtos listados sao vendidos em apenas 20% dos dias. Ou seja, mais de 75% dos dados sao compostos por mais de 80% de zeros. Isso definitivamente é um problema grave para o nosso modelo de regressao. 
# MAGIC 
# MAGIC Vamos tentar uma solucao nao-ideal para ver o que como o modelo se comporta. Vamos manter os produtos com poucos zeros, por exemplo com 80% de valores nao nulos e remover o restante

# COMMAND ----------

#Vamos pegar os produtos que venderam em 80% dos dias
freq_venda_min = 0.8*730

produtos_ = freq_prod[freq_prod['frequencia'] >= freq_venda_min]['produto_nome']
produtos_ = pd.DataFrame(produtos_)
produtos_

# COMMAND ----------

procentagem = 627/13734*100
print('mantemos apenas {:2f}% dos produtos'.format(procentagem))

# COMMAND ----------

prod_selecionados = produtos_.merge(corresp_prod)
prod_selecionados 

# COMMAND ----------



X_train = train_encod.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_train = train_encod["target_soma_movel_15"]

X_val = cv_encod.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_val = cv_encod["target_soma_movel_15"]

X_test = test3.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_test = test3["target_soma_movel_15"]

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### mudar os conjuntos de treino, teste o validacao com menos produtos

# COMMAND ----------

test_prod_red  = prod_selecionados.merge(test_encod.dropna()).drop(['produto_nome'], axis=1)
cv_prod_red    = prod_selecionados.merge(cv_encod).drop(['produto_nome'], axis=1)
train_prod_red = prod_selecionados.merge(train_encod).drop(['produto_nome'], axis=1)

# COMMAND ----------

test_prod_red.info()

# COMMAND ----------

X_train_prod_red = train_prod_red.drop(["venda_total", "target_soma_movel_15", "id_data" ], axis=1)
Y_train_prod_red = train_prod_red["target_soma_movel_15"]

X_val_prod_red = cv_prod_red.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_val_prod_red = cv_prod_red["target_soma_movel_15"]

X_test_prod_red = test_prod_red.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
Y_test_prod_red = test_prod_red["target_soma_movel_15"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinar modelo em conjunto de dados reduzido

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC regrXGB = xgb.XGBRegressor(booster='gbtree',    
# MAGIC                            n_estimators=2000,
# MAGIC                            early_stopping_rounds=50,
# MAGIC                            objective='reg:linear',
# MAGIC                            max_depth=2,    #arvores mais rasas para evitar overvitting
# MAGIC                            reg_lambda= 10, #acrescentar um L2 alto para ver se conseguimos evitar overvitting
# MAGIC                            learning_rate=0.008)

# COMMAND ----------

regrXGB.fit(X_train_prod_red, Y_train_prod_red, 
        eval_set=[(X_train_prod_red, Y_train_prod_red), (X_val_prod_red, Y_val_prod_red)],
        verbose=100)

# COMMAND ----------

#novas estatisticas
train_prod_red['target_soma_movel_15'].describe().reset_index()

# COMMAND ----------

# treino
print_metrics(X_train_prod_red, Y_train_prod_red)

# COMMAND ----------

# validacao
print_metrics(X_val_prod_red, Y_val_prod_red)

# COMMAND ----------

# MAGIC %md
# MAGIC Aparentemente, **conseguimos remover o overfitting** com essa solucao nao-ideal

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizar resultados para um produto

# COMMAND ----------

# a = float(corresp_prod[corresp_prod["produto_nome"] == "Produto 7273"]['Nproduto_nome'])

train_prod_red['previsao'] = regrXGB.predict(X_train_prod_red)
cv_prod_red['previsao']   = regrXGB.predict(X_val_prod_red)


train4_7273 = train_prod_red[train_prod_red['Nproduto_nome'] == a]
val4_7273   = cv_prod_red[cv_prod_red['Nproduto_nome'] == a]

#X_train4_7273 = train4_7273.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)
#X_val4_7273 = val4_7273.drop(["venda_total", "target_soma_movel_15", "id_data"], axis=1)


#train4_7273['previsao'] = regrXGB.predict(X_train4_7273)
#val4_7273['previsao']   = regrXGB.predict(X_val4_7273)

# COMMAND ----------

X_train_prod_red.columns

# COMMAND ----------

plot_resultados(train4_7273)

# COMMAND ----------

plot_resultados(val4_7273)

# COMMAND ----------

feature_importance= pd.DataFrame(data   = regrXGB.feature_importances_,
                 index  = regrXGB.feature_names_in_,
                 columns= ['importance'])

# COMMAND ----------

plt.figure(figsize=(10, 5))
fi = feature_importance.sort_values('importance', ascending=False).head(6)
fi.plot(kind='barh', title='Feature Importance')
plt.legend(loc='lower right')
plt.show()

# COMMAND ----------

#incluir lag de 1,3,4 dias

# COMMAND ----------

# MAGIC %md
# MAGIC # Demanda em 30, 60, e 90 dias no futuro
# MAGIC * Treinar novamente em todo o conjunto de *inteiro* (treino + teste)
# MAGIC * Criar dataframe vazio para os valores de datas futuras
# MAGIC * Rodar essas datas nos codigos de criaacao de feature

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinar novamente em todo o conjunto de *inteiro* (treino + teste)

# COMMAND ----------

todos = pd.concat([train_prod_red, cv_prod_red, test_prod_red])

# COMMAND ----------

X_todos = todos.drop(["venda_total", "target_soma_movel_15", "id_data", "previsao"], axis=1)
Y_todos = todos["target_soma_movel_15"]

# COMMAND ----------

regrXGB = xgb.XGBRegressor(booster='gbtree',    
                           n_estimators=2000,
                           objective='reg:linear',
                           max_depth=2,    #arvores mais rasas para evitar overvitting
                           reg_lambda= 10, #acrescentar um L2 alto para ver se conseguimos evitar overvitting
                           learning_rate=0.008)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC regrXGB.fit(X_todos, Y_todos,\
# MAGIC             eval_set = [(X_todos, Y_todos)], verbose=100)

# COMMAND ----------

Y_todos.describe()

# COMMAND ----------

print_metrics(X_todos, Y_todos)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar dataframe vazio para os valores de datas futuras

# COMMAND ----------

todos1

# COMMAND ----------

from datetime import timedelta

# COMMAND ----------

ultima_data_dados = todos1.id_data.max()
ultima_data_dados

# COMMAND ----------

added_date = pd.to_datetime(ultima_data_dados) + timedelta(days=90)
added_date = added_date.strftime("%Y-%m-%d")
added_date

# COMMAND ----------

#criando o data frame com as datas do futuro 
futuro    = pd.date_range("2019-12-17", '2020-03-30')
df_futuro = pd.DataFrame({'id_data':futuro})

# COMMAND ----------

df_futuro['futuro'] = True
todos1['futuro']     = False

# COMMAND ----------

futuro_datas_e_produtos = prod_selecionados.merge(df_futuro, how="cross")

# COMMAND ----------

futuro_datas_e_produtos

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos colocar esse dataframe futuro no fim do datafram existente para que possamos adicionar as features de lad corretamente

# COMMAND ----------

#todos2 = todos2.merge(corresp_prod, on='Nproduto_nome', how='left')
#todos2 = todos2.merge(corresp_ano, on='Ndiadoano', how='left')
#todos2 = todos2.merge(corresp_dmes, on='Ndiadomês', how='left')
#todos2 = todos2.merge(corresp_mes, on='Nmês', how='left')
#todos2 = todos2.merge(corresp_estacao, on='Nestação', how='left')
#todos2 = todos2.merge(corresp_sem_ano, on='Nsemanadoano', how='left')

# COMMAND ----------

todos1.columns

# COMMAND ----------

#todos2.drop(['previsao', 'produto_nome', 'diadoano', 'diadomês', 'mês', 'estação',\
#       'semanadoano'], axis=1, inplace=True)

#todos2

# COMMAND ----------

prod_selecionados

# COMMAND ----------

#cortar produtos

# COMMAND ----------

dados_prod_red = prod_selecionados.merge(todos1)

# COMMAND ----------

todos_e_futuro = pd.concat([dados_prod_red[dados_prod_red['id_data'] < "2019-12-17"], futuro_datas_e_produtos])

# COMMAND ----------

todos_e_futuro.reset_index(inplace=True)

# COMMAND ----------

todos_e_futuro = todos_e_futuro.sort_values(['Nproduto_nome', 'id_data']).reset_index()

# COMMAND ----------

todos_e_futuro.info()

# COMMAND ----------

todos_e_futuro.drop(['level_0', 'index'], axis=1, inplace=True)

# COMMAND ----------

todos_e_futuro

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rodar essas datas nos codigos de criacao de feature

# COMMAND ----------

#Vamos adicionar nossas features de tempo de lag e de rolling mean

# COMMAND ----------

todos_e_futuro = criar_features_de_tempo(todos_e_futuro)
todos_e_futuro

# COMMAND ----------

#encoding
#todos_e_futuro = todos_e_futuro.merge(corresp_prod, on='Nproduto_nome', how='left')
todos_e_futuro = todos_e_futuro.merge(corresp_ano, on='diadoano', how='left')
todos_e_futuro = todos_e_futuro.merge(corresp_dmes, on='diadomês', how='left')
todos_e_futuro = todos_e_futuro.merge(corresp_mes, on='mês', how='left')
todos_e_futuro = todos_e_futuro.merge(corresp_estacao, on='estação', how='left')
todos_e_futuro = todos_e_futuro.merge(corresp_sem_ano, on='semanadoano', how='left')
todos_e_futuro = todos_e_futuro.merge(corresp_semana, on='diadasemana', how='left')

# COMMAND ----------

todos_e_futuro

# COMMAND ----------

todos_e_futuro.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features de loop e rolling mean

# COMMAND ----------

lags = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]

# COMMAND ----------

def features_media_mov(df, janelas):
    for window in janelas:
        df['venda_media_movel_' + str(window)] = df.groupby(["produto_nome"])['venda_total']\
                                                 .transform(lambda x: x.shift(1)\
                                                 .rolling(window=window, min_periods=window)\
                                                 .mean())

# COMMAND ----------

def lag_features(df, lags):
    for lag in lags:
        df['lag_vendas_' + str(lag)] = df.groupby(["produto_nome"])['target_soma_movel_15']\
                                         .transform(lambda x: x.shift(lag))
    return df

# COMMAND ----------

lag_features

# COMMAND ----------

todos_e_futuro[todos_e_futuro['id_data'].between('2019-12-17', '2020-03-30')]

# COMMAND ----------

todos_e_futuro[todos_e_futuro['id_data'].between('2019-12-16', '2020-03-30')]

# COMMAND ----------

# MAGIC %%time
# MAGIC for data in df_futuro['id_data']:
# MAGIC     print(data)
# MAGIC     
# MAGIC     for lag in lags:
# MAGIC         #computar as features de lag para todos os dias
# MAGIC         todos_e_futuro1['lag_vendas_' + str(lag)] = todos_e_futuro1.groupby(["produto_nome"])['target_soma_movel_15']\
# MAGIC                                          .transform(lambda x: x.shift(lag))
# MAGIC         
# MAGIC     for window in janelas:
# MAGIC         #computar as features de media movel para todos os dias
# MAGIC         todos_e_futuro1['venda_media_movel_' + str(window)] = todos_e_futuro1.groupby(["produto_nome"])['venda_total']\
# MAGIC                                                  .transform(lambda x: x.shift(1)\
# MAGIC                                                  .rolling(window=window, min_periods=window)\
# MAGIC                                                  .mean())
# MAGIC     
# MAGIC     df_data = todos_e_futuro1[todos_e_futuro1['id_data']==data]
# MAGIC 
# MAGIC     #features
# MAGIC     X = df_data[['Nproduto_nome', 'lag_vendas_7', 'lag_vendas_14', 'lag_vendas_21',
# MAGIC        'lag_vendas_28', 'lag_vendas_35', 'lag_vendas_42', 'lag_vendas_49',
# MAGIC        'lag_vendas_56', 'lag_vendas_63', 'lag_vendas_70', 'lag_vendas_77',
# MAGIC        'lag_vendas_84', 'lag_vendas_91', 'venda_media_movel_7',
# MAGIC        'venda_media_movel_15', 'venda_media_movel_35', 'venda_media_movel_65',
# MAGIC        'venda_media_movel_95', 'venda_media_movel_120', 'Ndiadasemana',
# MAGIC        'Ndiadomês', 'Ndiadoano', 'Nmês', 'Nsemanadoano', 'Nestação']]
# MAGIC     
# MAGIC     
# MAGIC     #prever a target por dia
# MAGIC     pred = regrXGB.predict(X)
# MAGIC     
# MAGIC     #guardar a previsao
# MAGIC     df_data['target_soma_movel_15'] = pred
# MAGIC     
# MAGIC 
# MAGIC     
# MAGIC     # aproximar a venda total por dia pela soma movel das vendas dividido por 15 
# MAGIC     # esse valor sera utilizado para calcular a feature venda_media_movel_
# MAGIC     data_menos = pd.to_datetime(ultima_data_dados) - timedelta(days=90)
# MAGIC     data_menos = data_menos.strftime("%Y-%m-%d")
# MAGIC     
# MAGIC     df_data['venda_total'] = todos_e_futuro1[todos_e_futuro1['id_data']==data_menos]['target_soma_movel_15']/15
# MAGIC     
# MAGIC     
# MAGIC     #retirando a linha que nao tinha a target
# MAGIC     todos_e_futuro1 = todos_e_futuro1[todos_e_futuro1['id_data']!=data]
# MAGIC     
# MAGIC     #substituindo pela linha que tem a target
# MAGIC     todos_e_futuro1 = pd.concat([todos_e_futuro1, df_data]).sort_values(['Nproduto_nome', 'id_data'])\
# MAGIC                                                            .reset_index()\
# MAGIC                                                            .drop(['index'], axis=1)
# MAGIC     

# COMMAND ----------

# MAGIC %md
# MAGIC # Resultado da previsao de demanda em 30, 60 e 90 dias 

# COMMAND ----------

todos_e_futuro1[['target_soma_movel_15', 'Nproduto_nome', 'produto_nome', 'id_data']]

# COMMAND ----------

# MAGIC %md
# MAGIC A target é soma dos proximos 15 dias. Portanto, para prever a venda total em 30, 60 e 90 dias, podemos pegar:
# MAGIC ### Venda total em 60 dias:
# MAGIC vamos pegar a target_soma_movel_15 das datas:
# MAGIC * 2020-01-01
# MAGIC * 2020-01-16
# MAGIC 
# MAGIC ### Venda total em 60 dias:
# MAGIC vamos pegar a target_soma_movel_15 das datas:
# MAGIC * 2020-01-01
# MAGIC * 2020-01-16
# MAGIC * 2020-01-31
# MAGIC * 2020-02-15
# MAGIC 
# MAGIC ### Venda total em 90 dias:
# MAGIC vamos pegar a target_soma_movel_15 das datas:
# MAGIC * 2020-01-01
# MAGIC * 2020-01-16
# MAGIC * 2020-01-31
# MAGIC * 2020-02-15
# MAGIC * 2020-03-01
# MAGIC * 2020-03-16

# COMMAND ----------

primeiros_15_dias_por_produto = todos_e_futuro1[(todos_e_futuro1['id_data'] == '2020-01-01')]
primeiros_15_dias_por_produto = primeiros_15_dias_por_produto[['produto_nome', 'target_soma_movel_15']]

segundos_15_dias_por_produto = todos_e_futuro1[(todos_e_futuro1['id_data'] == '2020-01-16')]
segundos_15_dias_por_produto = segundos_15_dias_por_produto[['produto_nome', 'target_soma_movel_15']]

terceiros_15_dias_por_produto = todos_e_futuro1[(todos_e_futuro1['id_data'] == '2020-01-31')]
terceiros_15_dias_por_produto = terceiros_15_dias_por_produto[['produto_nome', 'target_soma_movel_15']]

quartos_15_dias_por_produto = todos_e_futuro1[(todos_e_futuro1['id_data'] == '2020-02-15')]
quartos_15_dias_por_produto = quartos_15_dias_por_produto[['produto_nome', 'target_soma_movel_15']]

quintos_15_dias_por_produto = todos_e_futuro1[(todos_e_futuro1['id_data'] == '2020-03-01')]
quintos_15_dias_por_produto = quintos_15_dias_por_produto[['produto_nome', 'target_soma_movel_15']]

sextos_15_dias_por_produto = todos_e_futuro1[(todos_e_futuro1['id_data'] == '2020-03-16')]
sextos_15_dias_por_produto = sextos_15_dias_por_produto[['produto_nome', 'target_soma_movel_15']]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Venda total por produto em 30 dias:

# COMMAND ----------

venda_total_30_dias = pd.DataFrame()
venda_total_30_dias['produto_nome'] = primeiros_15_dias_por_produto['produto_nome']

# COMMAND ----------

primeiros_15_dias_por_produto['target_soma_movel_15']

# COMMAND ----------

segundos_15_dias_por_produto['target_soma_movel_15'] + primeiros_15_dias_por_produto['target_soma_movel_15']

# COMMAND ----------

venda_total_30_dias['segundos_15_dias_por_produto'] = segundos_15_dias_por_produto['target_soma_movel_15']
venda_total_30_dias

# COMMAND ----------

venda_total_30_dias = pd.DataFrame()
venda_total_30_dias['produto_nome'] = primeiros_15_dias_por_produto.reset_index()['produto_nome']
venda_total_30_dias['numero_de_vendas'] = primeiros_15_dias_por_produto.reset_index()['target_soma_movel_15']\
                                            .add(segundos_15_dias_por_produto.reset_index()['target_soma_movel_15'], fill_value=0)
venda_total_30_dias

# COMMAND ----------

# MAGIC %md
# MAGIC ### Venda total por produto em 60 dias:

# COMMAND ----------

venda_total_60_dias = pd.DataFrame()
venda_total_60_dias['produto_nome'] = primeiros_15_dias_por_produto.reset_index()['produto_nome']
venda_total_60_dias['numero_de_vendas'] = venda_total_30_dias.reset_index()['numero_de_vendas'].add\
                                         (terceiros_15_dias_por_produto.reset_index()['target_soma_movel_15'], fill_value=0)

venda_total_60_dias['numero_de_vendas'] = venda_total_60_dias.reset_index()['numero_de_vendas'].add\
                                         (quartos_15_dias_por_produto.reset_index()['target_soma_movel_15'], fill_value=0)

venda_total_60_dias

# COMMAND ----------

venda_total_90_dias = pd.DataFrame()
venda_total_90_dias['produto_nome'] = primeiros_15_dias_por_produto.reset_index()['produto_nome']
venda_total_90_dias['numero_de_vendas'] = venda_total_60_dias.reset_index()['numero_de_vendas'].add\
                                         (quintos_15_dias_por_produto.reset_index()['target_soma_movel_15'], fill_value=0)

venda_total_90_dias['numero_de_vendas'] = venda_total_60_dias.reset_index()['numero_de_vendas'].add\
                                         (sextos_15_dias_por_produto.reset_index()['target_soma_movel_15'], fill_value=0)

venda_total_90_dias

# COMMAND ----------

# MAGIC %md
# MAGIC Assim conseguimos prever as demandas em 30, 60 e 90 para os 600 produtos mais vendidos!
