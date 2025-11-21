from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    average_precision_score, precision_score, make_scorer
)
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('dados.csv')

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

# MELHOR ESTRATÉGIA: Amostragem mais inteligente
# Manter TODAS as fraudes e fazer undersampling da classe majoritária
df_0 = df[df['is_fraud'] == 0]
df_1 = df[df['is_fraud'] == 1]

# Opção 1: Manter todas as fraudes e balancear
# Mantém todas as fraudes e pega proporção razoável da classe 0
n_fraudes = len(df_1)
n_nao_fraudes_manter = min(n_fraudes * 10, len(df_0))  # Proporção 10:1

df_0_sample = df_0.sample(n=n_nao_fraudes_manter, random_state=42)
df = pd.concat([df_0_sample, df_1], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

print("\n" + "="*60)
print("INFORMAÇÕES DO DATASET APÓS TRATAMENTO")
print("="*60)
print(f"\nTotal de registros: {len(df):,}")
print(f"\nDistribuição das classes:")
print(df['is_fraud'].value_counts())
print(f"\nProporção: {df['is_fraud'].value_counts(normalize=True)}")

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# =============================================================================
# OTIMIZAÇÃO DE HIPERPARÂMETROS
# =============================================================================

print("\n" + "="*60)
print("INICIANDO BUSCA DE HIPERPARÂMETROS")
print("="*60)

# Grid de hiperparâmetros para testar
param_distributions = {
    # Número de vizinhos: valores ímpares para evitar empates
    'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    
    # Peso dos vizinhos
    'model__weights': ['uniform', 'distance'],
    
    # Métrica de distância
    'model__metric': ['minkowski', 'euclidean', 'manhattan'],
    
    # Parâmetro p (1=Manhattan, 2=Euclidean)
    'model__p': [1, 2],
    
    # Parâmetros do SMOTE
    'smote__k_neighbors': [3, 5, 7],
    'smote__sampling_strategy': [0.5, 0.7, 1.0]  # Proporção de balanceamento
}

# Pipeline
pipeline = ImbPipeline([
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', KNeighborsClassifier(n_jobs=-1))
])

# Para fraudes, F1 ou Recall são mais importantes que Accuracy
# Vamos usar F1-Score como métrica principal
f1_scorer = make_scorer(f1_score)

# RandomizedSearchCV é mais rápido que GridSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=50,  # Número de combinações aleatórias a testar
    scoring={
        'f1': f1_scorer,
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    },
    refit='f1',  # Usar F1 para selecionar o melhor modelo
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("\nIniciando busca... (isso pode demorar)")
random_search.fit(X, y)

print("\n" + "="*60)
print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
print("="*60)
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

print("\n" + "="*60)
print("PERFORMANCE COM MELHORES HIPERPARÂMETROS")
print("="*60)
print(f"F1-Score (CV):  {random_search.cv_results_['mean_test_f1'][random_search.best_index_]:.4f}")
print(f"Recall (CV):    {random_search.cv_results_['mean_test_recall'][random_search.best_index_]:.4f}")
print(f"Precision (CV): {random_search.cv_results_['mean_test_precision'][random_search.best_index_]:.4f}")
print(f"ROC-AUC (CV):   {random_search.cv_results_['mean_test_roc_auc'][random_search.best_index_]:.4f}")

# =============================================================================
# AVALIAÇÃO FINAL COM 5-FOLD CV
# =============================================================================

print("\n" + "="*60)
print("AVALIAÇÃO FINAL COM 5-FOLD CROSS-VALIDATION")
print("="*60)

best_pipeline = random_search.best_estimator_
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': [],
    'pr_auc': []
}

fold = 1
for train_idx, test_idx in kfold.split(X, y):
    print(f"\nFOLD {fold}")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

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

    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC: {roc:.4f}")
    fold += 1

print("\n" + "="*60)
print("RESULTADOS FINAIS (MÉDIA ± DESVIO)")
print("="*60)
for metrica, valores in resultados.items():
    print(f"{metrica.upper():12} | {np.mean(valores):.4f} ± {np.std(valores):.4f}")

