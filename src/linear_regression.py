# Importação das bibliotecas necessárias para análise de dados, visualização e modelagem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
sns.set_theme()

# 1) CARREGAMENTO E INSPEÇÃO INICIAL DOS DADOS
# Lê o arquivo CSV. Ajuste o caminho conforme necessário.
raw_data = pd.read_csv('data/raw/1.04. Real life example - Data Science Bootcamp.csv')
print(raw_data.head())

# Remove a coluna 'Model' (neste dataset, costuma não ajudar no modelo e pode aumentar dimensionalidade)
data = raw_data.drop(['Model'],axis=1)

# Estatísticas descritivas para ter uma visão geral das distribuições e possíveis outliers
print(data.describe())

# Verifica valores nulos por coluna (importante para decidir estratégias de limpeza)
print(data.isnull().sum())

# 2) LIMPEZA BÁSICA
# Remove linhas com valores ausentes, seguindo a "Rule of thumb": 5% de dados ausentes por coluna
dataNoMV = data.dropna(axis=0)
dataNoMV.describe(include='all')

# Visualiza a distribuição do preço para identificar assimetria e outliers
sns.histplot(dataNoMV['Price'])

# Remove outliers extremos de preço (acima do percentil 99) para reduzir influência indevida no ajuste
q = dataNoMV['Price'].quantile(0.99)
data_1 = dataNoMV[dataNoMV['Price']<q]
data_1.describe(include='all')

# Confirma a distribuição do preço após o corte
sns.histplot(data_1['Price'])

# Remove outliers de 'Mileage' (acima do percentil 99), pois quilometragem muito alta pode distorcer o ajuste
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]

# Visualiza a quilometragem após o corte
sns.histplot(data_2['Mileage'], kde = True)

# Visualiza a distribuição de EngineV
sns.histplot(data_2['EngineV'], kde=True)

# Remove valores muito altos de EngineV (> 6.5), que parecem outliers no dataset
data_3 = data_2[data_2['EngineV']<6.5]

# Confere a distribuição de EngineV depois do filtro
sns.histplot(data_3['EngineV'], kde = True)

# Visualiza a distribuição de ano
sns.histplot(data_3['Year'], kde = True)

# Remove carros muito antigos (abaixo do percentil 1%), geralmente outliers em preço/condição
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]

# Confirma a distribuição de ano após corte inferior
sns.histplot(data_4['Year'], kde = True)

# Reseta o índice após remoções sucessivas
df_cln = data_4.reset_index(drop=True)
df_cln.describe(include='all')

# 3) ANÁLISE EXPLORATÓRIA: RELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS E O PREÇO
# Gráficos de dispersão para ver relações e possíveis linearidades
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))
ax1.scatter(df_cln['Year'],df_cln['Price'])
ax1.set_title('Price and Year')
ax2.scatter(df_cln['EngineV'],df_cln['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(df_cln['Mileage'],df_cln['Price'])
ax3.set_title('Price and Mileage')

plt.show()

# 4) TRANSFORMAÇÃO DO ALVO (PRICE) PARA LOG-PREÇO
# Transformação log reduz assimetria (skew) e pode linearizar relações, melhorando o ajuste da regressão
log_price = np.log(df_cln['Price'])
df_cln['log_price'] = log_price
print(df_cln)

# Visualiza a relação entre as features e o log do preço, tendo agora uma relação mais linear
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))
ax1.scatter(df_cln['Year'],df_cln['log_price'])
ax1.set_title('Price and Year')
ax2.scatter(df_cln['EngineV'],df_cln['log_price'])
ax2.set_title('Price and EngineV')
ax3.scatter(df_cln['Mileage'],df_cln['log_price'])
ax3.set_title('Price and Mileage')

plt.show()

# Remove a coluna Price original para trabalhar apenas com o log_price como alvo
df_cln = df_cln.drop(['Price'], axis = 1)

# Visualiza nomes de colunas (em notebook aparece; em script, use print)
df_cln.columns.values

# 5) MULTICOLINEARIDADE (VIF)
# VIF (Variance Inflation Factor) mede inflar variância por colinearidade. VIF alto indica colinearidade.
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df_cln[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
print(vif)

# Remove 'Year' por colinearidade (com Mileage, tipicamente). Isso simplifica o modelo e estabiliza coeficientes.
df_noMultiColl = df_cln.drop(['Year'],axis = 1)

# 6) VARIÁVEIS DUMMY
# Converte variáveis categóricas em dummies (0/1).
df_dummies = pd.get_dummies(df_noMultiColl, drop_first = True, dtype = int)
print(df_dummies.head())

# Visualiza as colunas após criação de dummies
print(df_dummies.columns)

# Reajuste na ordem das colunas para facilitar seleção futura
cols = ['log_price','Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
       'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
       'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
       'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
       'Registration_yes']

# Seleciona apenas as colunas de interesse (as que existirem no dataset)
df_preprocessed = df_dummies[cols]
print(df_preprocessed.head())

# 7) SEPARAÇÃO ENTRE ALVO (y) E FEATURES (x)
targets = df_preprocessed['log_price']
inputs = df_preprocessed.drop(['log_price'], axis = 1)

# 8) ESCALONAMENTO 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan

# 9) Separação treino/teste antes do scaler
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Escalonamento: ajusta o scaler apenas no conjunto de treino para evitar vazamento de dados
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled  = scaler.transform(x_test)

# 10) TREINAMENTO DO MODELO
reg = LinearRegression()
reg.fit(x_train,y_train)

# 11) AVALIAÇÃO EM TREINO
# Avaliação (treino)
y_hat_train = reg.predict(x_train_scaled)
print("R2 train:", r2_score(y_train, y_hat_train))
print("RMSE train:", np.sqrt(mean_squared_error(y_train, y_hat_train)))
print("MAE train:", mean_absolute_error(y_train, y_hat_train))

# Avaliação (teste)
y_hat_test = reg.predict(x_test_scaled)
print("R2 test:", r2_score(y_test, y_hat_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_hat_test)))
print("MAE test:", mean_absolute_error(y_test, y_hat_test))

# Diagnóstico de heterocedasticidade (Breusch-Pagan) usando resíduos do modelo em log-space
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(y_hat_test - y_test, sm.add_constant(x_test_scaled))
print("Breusch-Pagan p-value (LM):", lm_pvalue)

# Gráfico de Predição vs Alvo no conjunto de treino (em log-space)
plt.scatter(y_train, y_hat_train)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()

# Histograma dos resíduos (y_true - y_pred) em treino; deve ser aproximadamente simétrico ao redor de 0
sns.histplot(y_train - y_hat_train, kde = True)

# R² do treino (proporção da variância explicada).
r2_train = reg.score(x_train,y_train)
print(r2_train)
# Intercepto (bias) e coeficientes do modelo.
print(reg.intercept_)
print(reg.coef_)

# Resumo simples dos coeficientes em um DataFrame
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

# 12) AVALIAÇÃO EM TESTE
# Predições no conjunto de teste (em log-space)
y_hat_test = reg.predict(x_test)

# Converte de volta do log para o espaço original de preço (exp)
y_pred = np.exp(y_hat_test)
y_test_real = np.exp(y_test)

# Dispersão em log-space
fig, ax = plt.subplots()
ax.scatter(y_test, y_hat_test, alpha=0.2)
ax.set_xlabel('Targets (y_test)', fontsize=18)
ax.set_ylabel('Predictions (y_hat_test)', fontsize=18)
ax.set_xlim(6, 13)
ax.set_ylim(6, 13)

# Constrói DataFrame de comparação com predições e alvo no espaço original (preço)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns = ['Predictions'])
df_pf.head()

# Alinha índices de y_test (após train_test_split) para concatenar corretamente
y_test = y_test.reset_index(drop = True)
df_pf['Target'] = np.exp(y_test)
print(df_pf)

# Calcula resíduo (erro) no espaço original
df_pf['Residual'] = df_pf['Predictions'] - df_pf['Target']
print(df_pf)

# Ajustes de exibição
pd.options.display.max_rows = 50
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Calcula diferença percentual entre predição e alvo
df_pf['Difference%'] = (df_pf['Predictions'] - df_pf['Target']) / df_pf['Target'] * 100
df_pf.sort_values(by=['Difference%'])

# Gráfico Preço Real vs Preço Previsto (no espaço original), com linha y=x para referência
plt.figure(figsize=(6,6))
plt.scatter(y_test_real, y_pred, alpha=0.2)
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
plt.xlabel("Valor Real (Preço)")
plt.ylabel("Valor Previsto (Preço)")
plt.title("Previsão vs Valor Real")
plt.show()