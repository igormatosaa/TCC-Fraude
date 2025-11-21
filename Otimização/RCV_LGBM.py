from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dados.csv")

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

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline = ImbPipeline(steps=[
    ("scaler", RobustScaler()),
    ("smote", SMOTE(random_state=42)),
    ("lgbm", lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_jobs=-1,
        class_weight="balanced"
    ))
])

param_dist = {
    "lgbm__num_iterations": [100, 200, 500, 1000],
    "lgbm__max_depth": [10, 20, 30, -1],
    "lgbm__num_leaves": [31, 50, 100],
    "lgbm__min_data_in_leaf": [20, 50, 100],
    "lgbm__bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "lgbm__feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "lgbm__learning_rate": [0.01, 0.05, 0.1],
    "lgbm__lambda_l1": [0, 0.1, 0.5],
    "lgbm__lambda_l2": [0, 0.1, 0.5],
}

scorer = make_scorer(f1_score)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=6,
    scoring=scorer,
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("\nIniciando RandomizedSearch...\n")
random_search.fit(X_train, y_train)

print("\n====================================")
print("MELHORES PARÂMETROS ENCONTRADOS:")
print("====================================")
print(random_search.best_params_)

print("\n🔢 Melhor F1 médio (CV):")
print(random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== Desempenho no conjunto de teste ===")
print(classification_report(y_test, y_pred))

lgb_best = best_model.named_steps["lgbm"]

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": lgb_best.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== Top 20 características mais importantes ===")
print(importances.head(20))

plt.figure(figsize=(10,6))
sns.barplot(x="importance", y="feature", data=importances.head(20))
plt.title("Top 20 Features (LightGBM - Modelo Otimizado)")
plt.tight_layout()
plt.show()
