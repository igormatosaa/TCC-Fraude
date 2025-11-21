# === IMPORTAÇÕES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# === CARREGAMENTO DO DATASET ===
df = pd.read_csv('dados.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais:")
print(df.info())

# === TRATAMENTO INICIAL ===
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

# Remove 4 milhões de linhas
indices_para_remover = df.sample(n=4000000, random_state=42).index
df = df.drop(indices_para_remover)

print("\nPrimeiras linhas do dataset APÓS tratamento:")
print(df.head())
print("\nInformações APÓS tratamento:")
print(df.info())
print("\nDistribuição das classes APÓS tratamento:")
print(df['is_fraud'].value_counts())


# ================================
# === APLICANDO O SMOTE AQUI ===
# ================================

print("\nAplicando SMOTE...")

y = df['is_fraud']
X = df.drop(columns=['is_fraud'])

sm = SMOTE(random_state=42, sampling_strategy='auto')
X_res, y_res = sm.fit_resample(X, y)

print("\nDistribuição ANTES do SMOTE:")
print(y.value_counts(normalize=True) * 100)

print("\nDistribuição DEPOIS do SMOTE:")
print(y_res.value_counts(normalize=True) * 100)


ordem = ["0", "1"]
cores = {"0": "blue", "1": "orange"}

plt.figure(figsize=(12,5))

# === ANTES DO SMOTE ===
plt.subplot(1, 2, 1)
sns.countplot(
    x=y.astype(str),
    palette=cores,
    order=ordem      
)
plt.title("Antes do SMOTE")
plt.xlabel("is_fraud")

# === DEPOIS DO SMOTE ===
plt.subplot(1, 2, 2)
sns.countplot(
    x=y_res.astype(str),
    palette=cores,
    order=ordem     
)
plt.title("Depois do SMOTE")
plt.xlabel("is_fraud")

plt.tight_layout()
plt.show()



# =====================================
# === ANÁLISE EXPLORATÓRIA ORIGINAL ===
# =====================================

print("\nEstatísticas descritivas das colunas numéricas:")
print(X_res.describe().T)

num_cols = X_res.select_dtypes(include=['int64','float64']).columns.tolist()

# Histograma
plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2,3,i)
    sns.histplot(X_res[col], bins=50, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2,3,i)
    sns.boxplot(x=X_res[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Heatmap de correlação
corr = X_res[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de correlação entre variáveis numéricas')
plt.show()
