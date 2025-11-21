# === IMPORTAÇÕES ===
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns           

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, roc_auc_score,
    average_precision_score, precision_score
)
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# === CARREGAMENTO DO DATASET ===
df = pd.read_csv('dados.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações:")
print(df.info())
print("\nDistribuição das classes:")
print(df['is_fraud'].value_counts())

# === FUNÇÃO DE TRATAMENTO ===
def tratar(df):
    # Conversão de datas e cálculo da idade
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    dias_de_vida = (df['trans_date'] - df['dob']).dt.days
    idade_com_nan = dias_de_vida / 365.25
    
    mediana_idade = idade_com_nan.median()
    idade_com_nan.fillna(mediana_idade, inplace=True)
    df['idade'] = idade_com_nan.astype(int)

    # Colunas para remover
    colunas_para_remover = [
        'gender', 'city', 'state', 'zip', 'profile', 'merchant', 'ssn',
        'cc_num', 'first', 'last', 'street', 'acct_num', 'trans_num', 
        'job', 'dob', 'trans_date'
    ]
    df = df.drop(columns=colunas_para_remover)

    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])

    df['trans_time'] = pd.to_timedelta(df['trans_time']).dt.total_seconds().astype(int)
    
    return df

# === TRATAMENTO ===
df = tratar(df)

N = 4000000  # quantidade total a remover

# separa por classe
df_0 = df[df['is_fraud'] == 0]
df_1 = df[df['is_fraud'] == 1]

# tamanhos
n0 = len(df_0)
n1 = len(df_1)
total = n0 + n1

# proporção por classe
p0 = n0 / total
p1 = n1 / total

# quantidade a remover por classe
remove_0 = int(N * p0)
remove_1 = int(N * p1)

# seleciona linhas a remover
idx_remove_0 = df_0.sample(n=remove_0, random_state=42).index
idx_remove_1 = df_1.sample(n=remove_1, random_state=42).index

# junta e remove
df = df.drop(list(idx_remove_0) + list(idx_remove_1))

# Separação X / y
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print("\nInformações APÓS tratamento:")
print(df.info())
print("\nDistribuição das classes APÓS tratamento:")
print(df['is_fraud'].value_counts())

# 1. Definição do modelo base (igual)
model = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',
    n_jobs=1,
    metric='manhattan'
)

pipeline = ImbPipeline([ 
    ('scaler', RobustScaler()),    
    ('kbest', SelectKBest(k=5,score_func=f_classif)),
    ('knn', model)
])

# === STRATIFIED K-FOLD ===
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Guardar métricas
resultados = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': [],
    'pr_auc': []
}

# === EXECUTAR K-FOLD ===
fold = 1
for train_idx, test_idx in kfold.split(X, y):
    print(f"\n========== FOLD {fold} ==========")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    resultados['accuracy'].append(acc)
    resultados['precision'].append(prec)
    resultados['recall'].append(rec)
    resultados['f1'].append(f1)
    resultados['roc_auc'].append(roc)
    resultados['pr_auc'].append(pr)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")

    fold += 1

# === MÉDIA GERAL DAS MÉTRICAS ===
print("\n================== RESULTADOS FINAIS (MÉDIA DOS 5 FOLDS) ==================\n")
for metrica, valores in resultados.items():
    print(f"{metrica.upper():10} | MÉDIA = {np.mean(valores):.4f} | DESVIO = {np.std(valores):.4f}")

# === MATRIZ DE CORRELAÇÃO ===

print("\n=== MATRIZ DE CORRELAÇÃO ===")
corr = df.corr()

# Visualização (Heatmap)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de Correlação Completa")
plt.show()
