import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# === CARREGAMENTO DO DATASET ===
df = pd.read_csv('dados.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais:")
print(df.info())

def tratar(df):
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    dias_de_vida = (df['trans_date'] - df['dob']).dt.days
    idade_com_nan = dias_de_vida / 365.25
    idade_com_nan.fillna(idade_com_nan.median(), inplace=True)
    df['idade'] = idade_com_nan.astype(int)

    colunas_para_remover = [
        'gender', 'city', 'state', 'zip', 'profile', 'merchant', 'ssn',
        'cc_num', 'first', 'last', 'street', 'acct_num', 'trans_num',
        'job', 'dob', 'trans_date'
    ]
    df = df.drop(columns=colunas_para_remover)

    df = pd.get_dummies(df, columns=['category'], prefix='categoria', dtype=int)
    df['trans_time'] = pd.to_timedelta(df['trans_time']).dt.total_seconds().astype(int)
    return df

df = tratar(df)

indices_para_remover = df.sample(n=4000000, random_state=42).index
df = df.drop(indices_para_remover)

print("\nPrimeiras linhas do dataset APÓS tratamento:")
print(df.head())
print("\nInformações APÓS tratamento:")
print(df.info())
print("\nDistribuição das classes APÓS tratamento:")
print(df['is_fraud'].value_counts())

# === ANÁLISE EXPLORATÓRIA ===

### 1. Valores faltantes
# missing = df.isnull().sum().sort_values(ascending=False)
# missing_pct = (df.isnull().mean().sort_values(ascending=False) * 100).round(2)
# missing_df = pd.concat([missing, missing_pct], axis=1, keys=['count_missing','pct_missing'])
# print("\nValores faltantes por coluna:")
# print(missing_df[missing_df['count_missing'] > 0])



### 2. Distribuições e estatísticas descritivas
print("\nEstatísticas descritivas das colunas numéricas:")
print(df.describe().T)

# Histograma de algumas variáveis numéricas
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols[:6], 1):  # mostra somente as 6 primeiras para exemplo
    plt.subplot(2,3,i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

### 3. Verificar outliers com Boxplot
plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2,3,i)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

### 4. Correlações entre variáveis
corr = df[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de correlação entre variáveis numéricas')
plt.show()

### 5. Distribuição da variável-alvo
if 'is_fraud' in df.columns:
    print("\nDistribuição da variável-alvo (is_fraud):")
    print(df['is_fraud'].value_counts(normalize=True).round(4)*100)
    sns.countplot(x='is_fraud', data=df)
    plt.title('Contagem de fraudes vs não fraudes')
    plt.show()
else:
    print("A coluna ‘is_fraud’ não está presente no DataFrame.")

