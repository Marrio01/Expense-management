# app.py
from dash import Dash, dcc, html, Input, Output
import pandas as pd

from src.etl import (
    leer_excel,
    normalizar,
    agg_por_dia,
    agg_por_categoria,
    agg_por_semana,
    leer_resumen_semanal,
)
from src.graphics import (
    fig_pie_categorias,
    fig_serie_tiempo,
    fig_barras_semana,
    fig_serie_semana_pred,
    fig_serie_mensual_pred,
)
from src.model import (
    construir_dataset_semanal,
    predecir_proxima_semana,
    construir_dataset_mensual,
    predecir_proximo_mes,
)

DATA_PATH = "data/expenses.xlsx"

# Carga inicial
raw = leer_excel(DATA_PATH)
df = normalizar(raw)

# resumen semanal (presupuesto, gastado, disponible, ...)
df_sem = leer_resumen_semanal(DATA_PATH)

# dataset semanal y mensual histórico para el modelo
df_week_hist = construir_dataset_semanal(df)
pred_info = predecir_proxima_semana(df_week_hist)  # baseline semanal

df_month_hist = construir_dataset_mensual(df)
pred_month = predecir_proximo_mes(df_month_hist)   # baseline mensual

min_date = df["date"].min() if not df.empty else pd.Timestamp.today()
max_date = df["date"].max() if not df.empty else pd.Timestamp.today()

app = Dash(__name__)
app.title = "Control y Predicción de Gastos"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"},
    children=[
        html.H1("Control y Predicción de Gastos"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Rango de fechas"),
                        dcc.DatePickerRange(
                            id="rango",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format="DD/MM/YYYY",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Categoría"),
                        dcc.Dropdown(
                            id="filtro_cat",
                            options=[
                                {"label": c, "value": c}
                                for c in sorted(df["category"].dropna().unique())
                            ],
                            placeholder="Todas",
                            multi=True,
                        ),
                    ]
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
        ),

        # --- TARJETAS DE RESUMEN ---
        html.Div(
            id="resumen_cards",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, 1fr)",
                "gap": "12px",
                "marginTop": "20px",
            },
        ),

        # Selector de tipo de predicción
        html.Div(
            style={
                "marginTop": "20px",
                "marginBottom": "10px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "flex-start",
                "gap": "12px",
            },
            children=[
                html.Label("Tipo de predicción:"),
                dcc.RadioItems(
                    id="tipo_pred",
                    options=[
                        {"label": "Semanal", "value": "semanal"},
                        {"label": "Mensual", "value": "mensual"},
                    ],
                    value="semanal",
                    inline=True,
                ),
            ],
        ),

        html.Div(
            [
                dcc.Graph(id="pie_cat"),
                dcc.Graph(id="serie_dia"),
                dcc.Graph(id="barras_semana"),
                dcc.Graph(id="hist_pred"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "12px",
            },
        ),
    ],
)


@app.callback(
    [
        Output("pie_cat", "figure"),
        Output("serie_dia", "figure"),
        Output("barras_semana", "figure"),
        Output("resumen_cards", "children"),
        Output("hist_pred", "figure"),
    ],
    [
        Input("rango", "start_date"),
        Input("rango", "end_date"),
        Input("filtro_cat", "value"),
        Input("tipo_pred", "value"),
    ],
)
def actualizar(start_date, end_date, cats, tipo_pred):
    if df.empty or start_date is None or end_date is None:
        empty_cat = pd.DataFrame(columns=["category", "amount"])
        empty_day = pd.DataFrame(columns=["date", "amount"])
        empty_week = pd.DataFrame(columns=["week", "amount"])
        cards = []
        # figura vacía pero válida
        fig_empty_pred = fig_serie_semana_pred(
            pd.DataFrame(columns=["week", "spent"]), None
        )
        return (
            fig_pie_categorias(empty_cat),
            fig_serie_tiempo(empty_day),
            fig_barras_semana(empty_week),
            cards,
            fig_empty_pred,
        )

    # filtro por fechas
    f = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ]

    # filtro por categoría
    if cats:
        f = f[f["category"].isin(cats)]

    df_cat = agg_por_categoria(f)
    df_day = agg_por_dia(f)
    df_week = agg_por_semana(f)

    # --- RESUMEN NUMÉRICO PARA LAS TARJETAS ---

    total_gasto = float(f["amount"].sum())
    n_trans = int(len(f))
    gasto_medio_dia = float(df_day["amount"].mean()) if not df_day.empty else 0.0

    # semanas que caen en el rango filtrado
    weeks_sel = sorted(f["week"].unique())
    sem_sel = df_sem[df_sem["week"].isin(weeks_sel)]

    total_presupuesto = float(sem_sel["budget"].sum())
    total_disponible = float(sem_sel["available"].sum())
    total_ahorrado = float(sem_sel.get("saved", 0).sum())

    def tarjeta(titulo, valor, sufijo="€"):
        return html.Div(
            style={
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "padding": "10px 14px",
                "backgroundColor": "#f9f9f9",
            },
            children=[
                html.Div(titulo, style={"fontSize": "13px", "color": "#555"}),
                html.Div(
                    f"{valor:,.2f}{' ' + sufijo if sufijo else ''}"
                    if isinstance(valor, (int, float))
                    else str(valor),
                    style={"fontSize": "20px", "fontWeight": "bold"},
                ),
            ],
        )

    cards = [
        tarjeta("Gasto total (rango)", total_gasto),
        tarjeta("Presupuesto total (semanas del rango)", total_presupuesto),
        tarjeta("Disponible total (semanas del rango)", total_disponible),
        tarjeta("Ahorrado total (semanas del rango)", total_ahorrado),
    ]

    # tarjeta de predicción (siempre semanal por ahora)
    if pred_info is not None:
        titulo_pred = f"Predicción próxima semana ({pred_info['week_label']})"
        cards.append(tarjeta(titulo_pred, pred_info["prediction"]))

    # --- figura de predicción según selector ---
    if tipo_pred == "semanal":
        fig_pred = fig_serie_semana_pred(df_week_hist, pred_info)
    else:
        fig_pred = fig_serie_mensual_pred(df_month_hist, pred_month)

    return (
        fig_pie_categorias(df_cat),
        fig_serie_tiempo(df_day),
        fig_barras_semana(df_week),
        cards,
        fig_pred,
    )


if __name__ == "__main__":
    app.run(debug=True)
