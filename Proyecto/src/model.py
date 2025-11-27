# src/model.py
import pandas as pd
import datetime as dt
from typing import Optional, Dict

# mismo diccionario de abreviaturas de etl.py
MONTH_ABBR = {
    1: "ene", 2: "feb", 3: "mar", 4: "abr",
    5: "may", 6: "jun", 7: "jul", 8: "ago",
    9: "sept", 10: "oct", 11: "nov", 12: "dic",
}

def construir_dataset_semanal(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del dataframe normalizado (por transacción) construye
    un dataset semanal con el gasto total por semana.
    Supone que df_norm ya tiene week_idx, week, week_start (de etl.normalizar).
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["week_idx", "week", "week_start", "spent"])

    grp = (
        df_norm
        .groupby(["week_idx", "week", "week_start"], as_index=False)["amount"]
        .sum()
    )
    grp = grp.rename(columns={"amount": "spent"})
    grp = grp.sort_values("week_idx").reset_index(drop=True)
    return grp


def _make_week_label(start: dt.date, end: dt.date) -> str:
    """
    Genera una etiqueta tipo '17-22 ago' o '30-5 sept' según
    el formato que usas en tu Excel.
    """
    if start.month == end.month:
        label = f"{start.day}-{end.day} {MONTH_ABBR[start.month]}"
    else:
        # cuando cruza de mes, usamos solo el mes del final 
        label = f"{start.day}-{end.day} {MONTH_ABBR[end.month]}"
    return label


def predecir_proxima_semana(
    df_week: pd.DataFrame,
    ventana: int = 4
) -> Optional[Dict]:
    """
    Predicción baseline: promedio del gasto de las últimas `ventana` semanas.

    Devuelve un dict con:
    {
        "week_idx": int,
        "week_start": date,
        "week_end": date,
        "week_label": str,
        "prediction": float
    }
    o None si no hay datos suficientes.
    """
    if df_week.empty:
        return None

    df_week = df_week.sort_values("week_idx").reset_index(drop=True)

    # si hay pocas semanas, usamos todas las disponibles
    w = min(ventana, len(df_week))
    pred = df_week["spent"].tail(w).mean()

    last = df_week.iloc[-1]
    last_idx = int(last["week_idx"])
    last_start = pd.to_datetime(last["week_start"]).date()

    next_idx = last_idx + 1
    next_start = last_start + dt.timedelta(days=7)
    next_end = next_start + dt.timedelta(days=6)
    label = _make_week_label(next_start, next_end)

    return {
        "week_idx": next_idx,
        "week_start": next_start,
        "week_end": next_end,
        "week_label": label,
        "prediction": float(pred),
    }

# ===========================
#   MODELO MENSUAL
# ===========================

def construir_dataset_mensual(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del dataframe normalizado (por transacción),
    construye un dataset mensual con columnas:
    year, month, month_label, spent
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["year", "month", "month_label", "spent"])

    df = df_norm.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    month_map = {
        1: "ene", 2: "feb", 3: "mar", 4: "abr",
        5: "may", 6: "jun", 7: "jul", 8: "ago",
        9: "sept", 10: "oct", 11: "nov", 12: "dic",
    }

    df["month_label"] = df["month"].map(month_map)

    grp = (
        df.groupby(["year", "month", "month_label"], as_index=False)["amount"]
        .sum()
        .sort_values(["year", "month"])
        .rename(columns={"amount": "spent"})
        .reset_index(drop=True)
    )

    return grp


def predecir_proximo_mes(df_month: pd.DataFrame, ventana: int = 4):
    """
    Predicción baseline: promedio de últimos N meses.

    Retorna dict:
    {
        "year": int,
        "month": int,
        "label": "nov 2025",
        "prediction": float
    }
    """
    if df_month.empty:
        return None

    df_month = df_month.sort_values(["year", "month"]).reset_index(drop=True)

    w = min(ventana, len(df_month))
    pred = df_month["spent"].tail(w).mean()

    last = df_month.iloc[-1]
    last_year, last_month = int(last["year"]), int(last["month"])

    # siguiente mes
    if last_month == 12:
        next_year = last_year + 1
        next_month = 1
    else:
        next_year = last_year
        next_month = last_month + 1

    month_map = {
        1: "ene", 2: "feb", 3: "mar", 4: "abr",
        5: "may", 6: "jun", 7: "jul", 8: "ago",
        9: "sept", 10: "oct", 11: "nov", 12: "dic",
    }

    label = f"{month_map[next_month]} {next_year}"

    return {
        "year": next_year,
        "month": next_month,
        "label": label,
        "prediction": float(pred),
    }
