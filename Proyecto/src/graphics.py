# src/graphics.py
import plotly.express as px
import plotly.graph_objects as go

def fig_pie_categorias(df_cat):
    if df_cat.empty:
        return px.pie(values=[1], names=["Sin datos"], title="Gasto por categoría")
    return px.pie(df_cat, values="amount", names="category", title="Gasto por categoría")

def fig_serie_tiempo(df_day):
    if df_day.empty:
        # gráfico vacío pero válido
        return px.line(title="Gasto diario")
    return px.line(df_day, x="date", y="amount", title="Gasto diario")

def fig_barras_semana(df_week):
    if df_week.empty:
        return px.bar(title="Gasto por semana")
    return px.bar(df_week, x="week", y="amount", title="Gasto por semana")

# --- serie semanal histórica + punto de predicción --- #
def fig_serie_semana_pred(df_week_hist, pred_info=None):
    """
    df_week_hist: dataframe con columnas week, spent
    pred_info: dict devuelto por predecir_proxima_semana
    """
    fig = go.Figure()

    # barras históricas
    if df_week_hist is not None and not df_week_hist.empty:
        fig.add_bar(
            x=df_week_hist["week"],
            y=df_week_hist["spent"],
            name="Histórico",
        )

    # punto de predicción
    if pred_info is not None:
        fig.add_scatter(
            x=[pred_info["week_label"]],
            y=[pred_info["prediction"]],
            mode="markers+text",
            name="Predicción próxima semana",
            text=["Predicción"],
            textposition="top center",
        )

    fig.update_layout(
        title="Gasto semanal histórico y predicción",
        xaxis_title="Semana",
        yaxis_title="Gasto (€)",
    )

    return fig

def fig_serie_mensual_pred(df_month, pred_info=None):
    import plotly.graph_objects as go

    fig = go.Figure()

    if df_month is not None and not df_month.empty:
        labels = df_month["month_label"] + " " + df_month["year"].astype(str)
        fig.add_bar(
            x=labels,
            y=df_month["spent"],
            name="Histórico mensual",
        )

    if pred_info is not None:
        fig.add_scatter(
            x=[pred_info["label"]],
            y=[pred_info["prediction"]],
            mode="markers+text",
            name="Predicción mensual",
            text=["Predicción"],
            textposition="top center",
        )

    fig.update_layout(
        title="Gasto mensual histórico y predicción",
        xaxis_title="Mes",
        yaxis_title="Gasto (€)",
    )

    return fig
