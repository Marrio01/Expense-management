# Mario Godínez Chavero
#Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)

# Cargo datos
df = pd.read_csv("medical_insurance.csv")
print()

print("Primeras filas:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescripción estadística:")
print(df.describe(include="all"))

# Visualizaciones 

# Distribución coste anual
plt.figure()
sns.histplot(df["annual_medical_cost"], kde=True, bins=40)
plt.title("Distribución del coste médico anual")
plt.xlabel("annual_medical_cost")
plt.tight_layout()
plt.show()


# Coste médico anual segun high risk
if "is_high_risk" in df.columns:
    plt.figure()
    sns.boxplot(data=df, x="is_high_risk", y="annual_medical_cost")
    plt.title("Coste médico anual según is_high_risk")
    plt.xlabel("is_high_risk")
    plt.ylabel("annual_medical_cost")
    plt.tight_layout()
    plt.show()


# Coste médico vs edad
if "age" in df.columns:
    plt.figure()
    sns.scatterplot(
        data=df,
        x="age",
        y="annual_medical_cost", 
        hue="is_high_risk" if "is_high_risk" in df.columns else None,
        alpha=0.4
    )
    plt.title("Coste médico anual vs edad")
    plt.xlabel("Edad")
    plt.ylabel("annual_medical_cost")
    plt.tight_layout()
    plt.show()

# Coste médico vs IMC (bmi) 
if "bmi" in df.columns:
    plt.figure()
    sns.scatterplot(
        data=df,
        x="bmi",
        y="annual_medical_cost",
        hue="smoker" if "smoker" in df.columns else None,
        alpha=0.4
    )
    plt.title("Coste médico anual vs BMI (smoker)")
    plt.xlabel("BMI")
    plt.ylabel("annual_medical_cost")
    plt.tight_layout()
    plt.show()

# Coste medico vs número de enfermedades crónicas
if "chronic_count" in df.columns:
    chronic_cost = (
        df.groupby("chronic_count")["annual_medical_cost"]
        .mean()
        .reset_index()
    )
    plt.figure()
    sns.barplot(data=chronic_cost, x="chronic_count", y="annual_medical_cost")
    plt.title("Coste medio anual por número de enfermedades crónicas")
    plt.xlabel("chronic_count")
    plt.ylabel("Coste medio anual")
    plt.tight_layout()
    plt.show()


# 6 Coste total de reclamaciones vs días hospitalizado
if "days_hospitalized_last_3yrs" in df.columns and "total_claims_paid" in df.columns:
    plt.figure()
    sns.scatterplot(
        data=df,
        x="days_hospitalized_last_3yrs",
        y="total_claims_paid",
        alpha=0.4
    )
    plt.title("Total de reclamaciones pagadas vs días hospitalizado (3 años)")
    plt.xlabel("days_hospitalized_last_3yrs")
    plt.ylabel("total_claims_paid")
    plt.tight_layout()
    plt.show()

# Coste medio por tipo de plan
if "plan_type" in df.columns:
    plan_cost = (
        df.groupby("plan_type")["annual_medical_cost"]
        .mean()
        .reset_index()
    )

    plt.figure()
    sns.barplot(data=plan_cost, x="plan_type", y="annual_medical_cost")
    plt.title("Coste medio anual por tipo de plan")
    plt.xlabel("plan_type")
    plt.ylabel("Coste medio anual")
    plt.tight_layout()
    plt.show()

# Mapa de calor con correlación
candidate_num_cols = [
    "age", "bmi", "chronic_count", "risk_score",
    "annual_medical_cost", "total_claims_paid",
    "visits_last_year", "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs", "medication_count",
    "proc_surgery", "proc_imaging", "proc_consult_count",
    "deductible", "copay", "annual_premium", "monthly_premium",
    "claims_count", "avg_claim_amount"
]

num_cols = [c for c in candidate_num_cols if c in df.columns]

if len(num_cols) > 1:
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, fmt=".2f", square=True)
    plt.title("Matriz de correlación entre variables numéricas relevantes")
    plt.tight_layout()
    plt.show()

# 3. CLUSTERIZACIÓN (KMeans + PCA)

def run_clustering(dataframe, features, n_clusters=4):
    df_sub = dataframe[features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sub)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_out = dataframe.loc[df_sub.index].copy()
    df_out["cluster"] = clusters
    df_out["pca_1"] = X_pca[:, 0]
    df_out["pca_2"] = X_pca[:, 1]

    return df_out

cluster_features = [
    "age", "bmi", "chronic_count", "risk_score",
    "annual_medical_cost", "visits_last_year",
    "hospitalizations_last_3yrs", "medication_count"
]
cluster_features = [c for c in cluster_features if c in df.columns]

df_clustered = run_clustering(df, cluster_features, n_clusters=4)

# PCA coloreado por cluster 
plt.figure()
sns.scatterplot(
    data=df_clustered,
    x="pca_1", y="pca_2",
    hue="cluster",
    palette="tab10",
    alpha=0.6
)
plt.title("Clusters de pacientes (PCA 2D)")
plt.tight_layout()
plt.show()

# Gráfico cluster 2: coste médico por cluster
plt.figure()
sns.boxplot(data=df_clustered, x="cluster", y="annual_medical_cost")
plt.title("Coste médico anual por cluster de pacientes")
plt.tight_layout()
plt.show()

# Resumen de clusters 
cluster_summary = df_clustered.groupby("cluster")[
    ["annual_medical_cost", "risk_score", "chronic_count", "visits_last_year"]
].mean()
print("\nResumen medio de variables por cluster:")
print(cluster_summary)

# MODELO PREDICTIVO
# CLASIFICACIÓN: is_high_risk

# variable  0/1
if df["is_high_risk"].dtype == bool:
    df["is_high_risk_binary"] = df["is_high_risk"].astype(int)
else:
    # Mapeo genérico
    mapping = {
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "high": 1, "low": 0,
        True: 1, False: 0,
        1: 1, 0: 0
    }
    df["is_high_risk_binary"] = df["is_high_risk"].map(mapping)

# features numéricas
model_features = [
    "age", "bmi", "chronic_count", "risk_score",
    "visits_last_year", "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs",
    "medication_count",
    "annual_premium", "deductible", "copay",
    "claims_count", "avg_claim_amount", "total_claims_paid"
]
model_features = [f for f in model_features if f in df.columns]

df_model = df.dropna(subset=["is_high_risk_binary"] + model_features).copy()

X = df_model[model_features].astype(float)
y = df_model["is_high_risk_binary"].astype(int)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n RESULTADOS MODELO CLASIFICACION is_high_risk")
print("Classification report:")
print(classification_report(y_test, y_pred))

try:
    roc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc:.3f}")
except Exception:
    pass

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

coef_df = pd.DataFrame({
    "variable": model_features,
    "coeficiente": log_reg.coef_[0]
}).sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)

print("\nCoeficientes del modelo:")
print(coef_df.to_string(index=False))

""""
- Variables con coeficiente positivo:
  cuanto más alto su valor, mayor probabilidad de ser clasificado como high risk.
- Variables con coeficiente negativo:
  cuanto más alto su valor, menor probabilidad de high risk.
"""
