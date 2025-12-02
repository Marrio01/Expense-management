# DASHBOARD 
# 3 GRÁFICAS 
# 1 GRAFICA DE COEFICIENTES

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# CARGA Y PREPROCESADO
df = pd.read_csv("medical_insurance.csv")

# Aseguramos que is_high_risk es 0/1
if df["is_high_risk"].dtype == bool:
    df["is_high_risk_binary"] = df["is_high_risk"].astype(int)
else:
    mapping = {
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "high": 1, "low": 0,
        True: 1, False: 0,
        1: 1, 0: 0
    }
    df["is_high_risk_binary"] = df["is_high_risk"].map(mapping)

# FEATURES PARA EL MODELO 
model_features = [
    "age", "bmi", "chronic_count",
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_scaled, y)

coef_df = pd.DataFrame({
    "variable": model_features,
    "coeficiente": log_reg.coef_[0]
}).sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)

# Figura de coeficientes
coef_fig = px.bar(
    coef_df,
    x="variable",
    y="coeficiente",
    title="Coeficientes del modelo de clasificación (is_high_risk)",
)
coef_fig.update_layout(xaxis_tickangle=-45)

# Dash app
app = Dash(__name__)

# Opciones de región 
if "region" in df.columns:
    regions = sorted(df["region"].dropna().unique().tolist())
else:
    regions = []

region_options = [{"label": "Todas", "value": "Todas"}] + [
    {"label": r, "value": r} for r in regions
]

app.layout = html.Div([
    html.H1("Dashboard Costes Médicos y Riesgo Clínico"),

    html.Div([
        html.Label("Filtrar por región"),
        dcc.Dropdown(
            id="region_dropdown",
            options=region_options,
            value="Todas",
            clearable=False
        )
    ], style={"width": "40%", "marginBottom": "20px"}),

    html.Div([
        dcc.Graph(id="cost_age_scatter"),     # 1) Coste vs edad
        dcc.Graph(id="cost_by_risk_box"),     # 2) coste vs is_high_risk
    ], style={"display": "flex", "flexWrap": "wrap"}),

    html.Div([
        dcc.Graph(id="chronic_cost_bar"),     # 3) Coste medio por chronic_count
        dcc.Graph(id="coef_bar", figure=coef_fig),  # 4) Coeficientes del modelo
    ], style={"display": "flex", "flexWrap": "wrap"}),
])

# Callbacks
@app.callback(
    Output("cost_age_scatter", "figure"),
    Output("cost_by_risk_box", "figure"),
    Output("chronic_cost_bar", "figure"),
    Input("region_dropdown", "value"),
)
def update_graphs(selected_region):

    if selected_region != "Todas" and "region" in df.columns:
        dff = df[df["region"] == selected_region].copy()
    else:
        dff = df.copy()

    # 1) Scatter coste vs edad
    if "age" in dff.columns and "annual_medical_cost" in dff.columns:
        fig_scatter = px.scatter(
            dff,
            x="age",
            y="annual_medical_cost",
            color="is_high_risk" if "is_high_risk" in dff.columns else None,
            title=f"Coste médico anual vs edad (Región: {selected_region})",
            opacity=0.5
        )
    else:
        fig_scatter = px.scatter(title="Faltan columnas age o annual_medical_cost")

    # 2) Boxplot coste vs is_high_risk
    if "is_high_risk" in dff.columns and "annual_medical_cost" in dff.columns:
        fig_box = px.box(
            dff,
            x="is_high_risk",
            y="annual_medical_cost",
            title=f"Coste anual según is_high_risk (Región: {selected_region})"
        )
    else:
        fig_box = px.box(title="Faltan columnas is_high_risk o annual_medical_cost")

    # 3) Coste medio por chronic_count
    if "chronic_count" in dff.columns and "annual_medical_cost" in dff.columns:
        chronic_cost = (
            dff.groupby("chronic_count")["annual_medical_cost"]
            .mean()
            .reset_index()
        )
        fig_bar_chronic = px.bar(
            chronic_cost,
            x="chronic_count",
            y="annual_medical_cost",
            title=f"Coste medio anual por número de enfermedades crónicas (Región: {selected_region})"
        )
    else:
        fig_bar_chronic = px.bar(title="Faltan columnas chronic_count o annual_medical_cost")

    return fig_scatter, fig_box, fig_bar_chronic


if __name__ == "__main__":
    app.run(debug=True)
