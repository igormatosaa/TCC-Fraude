from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgbm
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, make_scorer, RocCurveDisplay, roc_auc_score,
    accuracy_score, recall_score,precision_score, average_precision_score, confusion_matrix,
    classification_report
)
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

    df = pd.get_dummies(df, columns=['category'], prefix='categoria', dtype=int, drop_first=True)
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

# Normalização de colunas numéricas
# scaler = RobustScaler()
# cols_norm = ['lat', 'long', 'city_pop', 'unix_time', 'amt', 'merch_lat', 'merch_long']
# X[cols_norm] = scaler.fit_transform(X[cols_norm])

# === DIVISÃO TREINO / TESTE ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# === FUNÇÃO DE AVALIAÇÃO (VALIDAÇÃO CRUZADA) ===
def avaliar_modelo_com_cv(model, X, y, n_splits=5):
    # Dicionário de métricas atualizado para incluir 'precision'
    metrics = {
        'accuracy': [], 
        'precision': [], # Adicionado Precision
        'f1_score': [], 
        'recall': [], 
        'specificity': [], 
        'roc_auc': [], 
        'pr_auc': []
    }
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"--- Iniciando Validação Cruzada ({n_splits} folds) ---")
    
    # 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Treinamento e Previsão
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Cálculo da Specificity (que requer a matriz de confusão)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Armazenamento das métricas
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred)) # Calculando e armazenando Precision
        metrics['f1_score'].append(f1_score(y_val, y_pred))
        metrics['recall'].append(recall_score(y_val, y_pred))
        metrics['specificity'].append(specificity)
        metrics['roc_auc'].append(roc_auc_score(y_val, y_proba))
        metrics['pr_auc'].append(average_precision_score(y_val, y_proba))
        
        print(f" -> Fold {fold+1} concluído.")

    # Geração do DataFrame de resultados
    results_df = pd.DataFrame(index=metrics.keys())
    results_df['Média'] = [np.mean(v) for v in metrics.values()]
    results_df['Desvio Padrão'] = [np.std(v) for v in metrics.values()]

    print("\n--- Resultados da Validação Cruzada ---")
    print(results_df)
    return results_df

# --- PIPELINE BASE ---


# Criando o modelo com os parâmetros
# Adicionei random_state para reprodutibilidade e n_jobs=-1 para velocidade
model = lgbm.LGBMClassifier(
    subsample=0.9,
    reg_lambda=0,
    reg_alpha=0,
    num_leaves=100,
    n_estimators=1000,
    min_child_samples=20,
    max_depth=30, 
    learning_rate=0.1,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
pipeline = ImbPipeline([
     ('smote', SMOTE(random_state=42)),
     ('lightgbm', model)
])


# 2. Passe os parâmetros para o .fit()
#pipeline.fit(X_train, y_train)
avaliar_modelo_com_cv(pipeline,X, y)
# --- AVALIAÇÃO FINAL NO TESTE ---
# best_model = pipeline
# y_pred = best_model.predict(X_test)
# y_proba_test = best_model.predict_proba(X_test)[:, 1]

# from sklearn.metrics import classification_report, confusion_matrix
# print("\n=== Desempenho no conjunto de teste ===")
# print(classification_report(y_test, y_pred))

# # --- IMPORTÂNCIA DAS FEATURES ---
# rf_best = best_model.named_steps['lightgbm']

# importances = pd.DataFrame({
#     'feature': X.columns,
#     'importance': rf_best.feature_importances_
# }).sort_values(by='importance', ascending=False)

# print("\n=== 20 Variáveis Mais Importantes ===")
# print(importances.head(20))

# plt.figure(figsize=(10,6))
# sns.barplot(x='importance', y='feature', data=importances.head(20), palette='viridis')
# plt.title('Top 20 Variáveis Mais Relevantes (Random Forest - Melhor Configuração)')
# plt.xlabel('Importância')
# plt.ylabel('Variável')
# plt.tight_layout()
# plt.show()

# # --- PLOTAGEM DA CURVA ROC ---
# print("\n=== Plotando Curva ROC ===")
# auc_final = roc_auc_score(y_test, y_proba_test)

# plt.figure(figsize=(8, 8))
# # Gera a curva ROC diretamente dos dados de teste e probabilidades
# roc_display = RocCurveDisplay.from_predictions(
#     y_test, 
#     y_proba_test, 
#     name=f"LightGBM (AUC: {auc_final:.4f})",
#     ax=plt.gca()
# )

# # Adiciona a linha de referência (modelo aleatório)
# plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
# plt.title('Curva ROC no Conjunto de Teste')
# plt.xlabel('Taxa de Falsos Positivos (False Positive Rate)')
# plt.ylabel('Taxa de Verdadeiros Positivos (True Positive Rate)')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
# print("Curva ROC plotada com sucesso.")