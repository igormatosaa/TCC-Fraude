from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('dados.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações:")
print(df.info())
print("\nDistribuição das classes:")
print(df['is_fraud'].value_counts())

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

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

scaler = RobustScaler()
cols_norm = ['lat', 'long', 'city_pop', 'unix_time', 'amt', 'merch_lat', 'merch_long']
X[cols_norm] = scaler.fit_transform(X[cols_norm])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('random_forest', rf)
])

param_dist = {
    'random_forest__bootstrap': [True, False],
    'random_forest__criterion': ['gini', 'entropy'],
    'random_forest__max_depth': [1, 2, 4, 8, 16, 32, None],
    'random_forest__max_features': [2, 4, 6, 8, 10, 12, 15, 20],
    'random_forest__max_leaf_nodes': [10, 100, 500, None],
    'random_forest__min_impurity_decrease': [0, 0.001, 0.005, 0.01, 0.05],
    'random_forest__min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 128],
    'random_forest__min_samples_split': [2, 4, 8, 16, 32, 64, 128],
    'random_forest__n_estimators': [100, 150, 200, 250, 500]
}

scorer = make_scorer(f1_score, average='binary')

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=300,            
    scoring=scorer,
    cv=5,                 
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("Iniciando busca aleatória de hiperparâmetros...")
random_search.fit(X_train, y_train)

print("\nMelhor combinação de parâmetros encontrada:")
print(random_search.best_params_)

print("\nMelhor pontuação média de F1:")
print(random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import classification_report
print("\n=== Desempenho no conjunto de teste ===")
print(classification_report(y_test, y_pred))

rf_best = best_model.named_steps['random_forest']
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n=== 20 Variáveis Mais Importantes ===")
print(importances.head(20))

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=importances.head(20), palette='viridis')
plt.title('Top 20 Variáveis Mais Relevantes (Random Forest - Melhor Configuração)')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.show()