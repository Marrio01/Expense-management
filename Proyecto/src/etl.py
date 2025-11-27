# src/etl.py
import pandas as pd
import datetime as dt

DATE_COLS  = ["Fecha", "fecha", "date"]
CAT_COLS   = ["Categoría", "Categoria", "category"]
ITEM_COLS  = ["Producto", "producto", "item", "concepto"]
STORE_COLS = ["Tienda", "tienda", "store"]

AMOUNT_COLS = [
    "Precio (€)",
    "Precio $",
    "Precio ($)",
    "Precio total",
    "Total",
    "Importe",
    "Monto",
]

def _first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def leer_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    date_c  = _first_existing(df, DATE_COLS)
    cat_c   = _first_existing(df, CAT_COLS)
    item_c  = _first_existing(df, ITEM_COLS)
    store_c = _first_existing(df, STORE_COLS)
    amt_c   = _first_existing(df, AMOUNT_COLS)

    out = pd.DataFrame()

    # fecha
    if date_c:
        out["date"] = pd.to_datetime(df[date_c], dayfirst=True, errors="coerce")

    # categoría rellenando hacia abajo (para quitar NaN de bloques)
    if cat_c:
        cat_series = df[cat_c].ffill().fillna("Sin categoría")
        out["category"] = cat_series.astype(str)

    # otros campos
    if item_c:
        out["item"] = df[item_c].astype(str)
    if store_c:
        out["store"] = df[store_c].astype(str)

    # importe
    if amt_c:
        out["amount"] = pd.to_numeric(df[amt_c], errors="coerce")
    else:
        out["amount"] = 0.0

    # limpiar filas sin fecha
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    out["amount"] = out["amount"].fillna(0.0)

    # ---------- semanas organizadas como tu hoja ----------

    # fecha mínima: donde empieza la primera semana (17/08/2025 en tu caso)
    first_date = out["date"].min().date()

    # helper para calcular índice de semana y rango (inicio/fin)
    def _week_idx_and_range(d: pd.Timestamp):
        d_date = d.date()

        base0_start = first_date                  # 17/08
        base0_end   = first_date + dt.timedelta(days=5)  # 17 + 5 -> 22/08

        if d_date <= base0_end:
            idx = 0
            start, end = base0_start, base0_end
        else:
            base1_start = base0_end + dt.timedelta(days=1)  # 23/08
            days_from_base1 = (d_date - base1_start).days
            idx = 1 + days_from_base1 // 7
            start = base1_start + dt.timedelta(days=(idx - 1) * 7)
            end = start + dt.timedelta(days=6)
        return idx, start, end

    week_idx_list = []
    week_start_list = []
    week_end_list = []

    for d in out["date"]:
        idx, ws, we = _week_idx_and_range(d)
        week_idx_list.append(idx)
        week_start_list.append(ws)
        week_end_list.append(we)

    out["week_idx"] = week_idx_list
    out["week_start"] = week_start_list
    out["week_end"] = week_end_list

    # etiquetas tipo "17-22 ago", "30-5 sept", etc.
    month_abbr = {
        1: "ene", 2: "feb", 3: "mar", 4: "abr",
        5: "may", 6: "jun", 7: "jul", 8: "ago",
        9: "sept", 10: "oct", 11: "nov", 12: "dic",
    }

    labels = []
    for s, e in zip(out["week_start"], out["week_end"]):
        if s.month == e.month:
            label = f"{s.day}-{e.day} {month_abbr[s.month]}"
        else:
            # solo mostramos el mes del final, como en tu Excel: "30-5 sept"
            label = f"{s.day}-{e.day} {month_abbr[e.month]}"
        labels.append(label)

    out["week"] = labels

    # ------------------------------------------------------

    return out

def agg_por_dia(df_norm: pd.DataFrame) -> pd.DataFrame:
    if df_norm.empty:
        return pd.DataFrame(columns=["date", "amount"])
    return (
        df_norm.groupby("date", as_index=False)["amount"]
        .sum()
        .sort_values("date")
    )

def agg_por_categoria(df_norm: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    f = df_norm.copy()
    if start is not None:
        f = f[f["date"] >= pd.to_datetime(start)]
    if end is not None:
        f = f[f["date"] <= pd.to_datetime(end)]

    if f.empty:
        return pd.DataFrame(columns=["category", "amount"])

    return (
        f.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )

def agg_por_semana(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por semana respetando el orden de week_idx
    y usando la etiqueta 'week' (17-22 ago, 23-29 ago, ...).
    """
    if df_norm.empty:
        return pd.DataFrame(columns=["week", "amount"])

    tmp = (
        df_norm.groupby(["week_idx", "week"], as_index=False)["amount"]
        .sum()
        .sort_values("week_idx")
    )
    return tmp[["week", "amount"]]


# --- ETL para resumen semanal (presupuesto, gastado, disponible, etc.) ---

WEEK_LABEL_COLS = ["Semana", "week"]
BUDGET_COLS     = ["Presupuesto (€)", "Presupuesto", "Budget"]
SPENT_COLS      = ["Gastado (€)", "Gastado", "Spent"]
REC_COLS        = ["Gasto Recurrente Esperado(€)", "Gasto recurrente esperado"]
PROJ_COLS       = ["Total proyectado (€)", "Total proyectado"]
AVAIL_COLS      = ["Disponible (€)", "Disponible"]
SAVED_COLS      = ["Ahorrado (€)", "Ahorrado"]

def leer_resumen_semanal(path: str) -> pd.DataFrame:
    """
    Lee la hoja semanal del mismo Excel (segunda hoja) y
    la normaliza a columnas:
    week, budget, spent, expected_recurring, projected, available, saved
    """
    # suponemos que la hoja semanal es la segunda hoja (índice 1)
    df = pd.read_excel(path, sheet_name=1)

    week_c  = _first_existing(df, WEEK_LABEL_COLS)
    bud_c   = _first_existing(df, BUDGET_COLS)
    spent_c = _first_existing(df, SPENT_COLS)
    rec_c   = _first_existing(df, REC_COLS)
    proj_c  = _first_existing(df, PROJ_COLS)
    avail_c = _first_existing(df, AVAIL_COLS)
    saved_c = _first_existing(df, SAVED_COLS)

    out = pd.DataFrame()

    if week_c:
        out["week"] = df[week_c].astype(str)
    else:
        return pd.DataFrame(columns=["week","budget","spent","expected_recurring","projected","available","saved"])

    if bud_c:
        out["budget"] = pd.to_numeric(df[bud_c], errors="coerce").fillna(0.0)
    else:
        out["budget"] = 0.0

    if spent_c:
        out["spent"] = pd.to_numeric(df[spent_c], errors="coerce").fillna(0.0)
    else:
        out["spent"] = 0.0

    if rec_c:
        out["expected_recurring"] = pd.to_numeric(df[rec_c], errors="coerce").fillna(0.0)
    else:
        out["expected_recurring"] = 0.0

    if proj_c:
        out["projected"] = pd.to_numeric(df[proj_c], errors="coerce").fillna(0.0)
    else:
        out["projected"] = 0.0

    if avail_c:
        out["available"] = pd.to_numeric(df[avail_c], errors="coerce").fillna(0.0)
    else:
        out["available"] = 0.0

    if saved_c:
        out["saved"] = pd.to_numeric(df[saved_c], errors="coerce").fillna(0.0)
    else:
        out["saved"] = 0.0

    # quitar filas vacías de semana
    out = out[out["week"].str.strip() != ""].reset_index(drop=True)

    return out
