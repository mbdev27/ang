# app_unico.py
# Aplicação Streamlit completa em um único arquivo
# Calculadora de Ângulos e Distâncias – UFPE (método das direções)

import io
import math
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =============================================================================
# 1. FUNÇÕES AUXILIARES – IDENTIFICAÇÃO (equivalente ao utils.py)
# =============================================================================

def _parse_data_flex(valor):
    """
    Tenta interpretar 'valor' como data e devolver string no formato DD/MM/AAAA.
    Retorna string vazia se não conseguir interpretar.
    """
    if pd.isna(valor):
        return ""

    if isinstance(valor, (datetime, pd.Timestamp)):
        return valor.strftime("%d/%m/%Y")

    s = str(valor).strip()
    if s == "":
        return ""

    formatos = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%y",
        "%d-%m-%y",
    ]
    for fmt in formatos:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return ""


def ler_identificacao_from_df(df_id: pd.DataFrame):
    """
    Lê a aba 'Identificação' do Excel em formato flexível:

        Campo | Valor
        ------+------
        Professor(a) | ...
        Equipamento  | ...
        Dados        | ...
        Local        | ...
        Patrimônio   | ...

    Retorna dicionário:
      'Professor(a)', 'Equipamento', 'Dados', 'Local', 'Patrimônio'

    Em que 'Dados' é string 'DD/MM/AAAA' ou ''.
    """
    info = {
        "Professor(a)": "",
        "Equipamento": "",
        "Dados": "",
        "Local": "",
        "Patrimônio": "",
    }

    if df_id is None or df_id.empty:
        return info

    cols_lower = [str(c).strip().lower() for c in df_id.columns]
    col_campo = None
    col_valor = None
    for i, c in enumerate(cols_lower):
        if c in ["campo", "campos", "descricao", "descrição", "item"]:
            col_campo = df_id.columns[i]
        if c in ["valor", "valores", "dado"]:
            col_valor = df_id.columns[i]

    if col_campo is None:
        col_campo = df_id.columns[0]
    if col_valor is None:
        col_valor = df_id.columns[1] if len(df_id.columns) > 1 else df_id.columns[0]

    for _, row in df_id.iterrows():
        campo = str(row.get(col_campo, "")).strip().lower()
        valor = row.get(col_valor, "")

        if "professor" in campo:
            info["Professor(a)"] = str(valor).strip()
        elif "equip" in campo:
            info["Equipamento"] = str(valor).strip()
        elif campo in ["data", "dados", "data dos dados", "data da atividade"]:
            info["Dados"] = _parse_data_flex(valor)
        elif "local" in campo:
            info["Local"] = str(valor).strip()
        elif ("patrim" in campo) or ("tomb" in campo):
            info["Patrimônio"] = str(valor).strip()

    return info


# =============================================================================
# 2. FUNÇÕES DE PROCESSAMENTO (equivalente ao processing.py)
# =============================================================================

REQUIRED_COLS_BASE = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]
OPTIONAL_COLS = ["SEQ"]
REQUIRED_COLS_ALL = REQUIRED_COLS_BASE + OPTIONAL_COLS


# --------------------------- Ângulos ---------------------------

def parse_angle_to_decimal(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "":
        return float("nan")

    try:
        if all(ch.isdigit() or ch in ".,-+" for ch in s):
            return float(s.replace(",", "."))
    except Exception:
        pass

    for ch in ["°", "º", "'", "´", "′", '"', "″"]:
        s = s.replace(ch, " ")
    s = s.replace(",", ".")
    partes = [p for p in s.split() if p != ""]
    if not partes:
        return float("nan")

    try:
        deg = float(partes[0])
        minutos = float(partes[1]) if len(partes) > 1 else 0.0
        segundos = float(partes[2]) if len(partes) > 2 else 0.0
    except Exception:
        return float("nan")

    sinal = 1.0
    if deg < 0:
        sinal = -1.0
        deg = abs(deg)
    return sinal * (deg + minutos / 60.0 + segundos / 3600.0)


def decimal_to_dms(angle_deg: float) -> str:
    if angle_deg is None or math.isnan(angle_deg):
        return ""
    a = angle_deg % 360.0
    d = int(a)
    m_f = (a - d) * 60
    m = int(m_f)
    s_f = (m_f - m) * 60
    s = int(round(s_f))
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d += 1
    return f"{d:02d}°{m:02d}'{s:02d}\""


def mean_direction_circular(angles_deg: List[float]) -> float:
    vals = [a for a in angles_deg if not math.isnan(a)]
    if len(vals) == 0:
        return float("nan")
    x = sum(math.cos(math.radians(v)) for v in vals)
    y = sum(math.sin(math.radians(v)) for v in vals)
    if x == 0 and y == 0:
        return float("nan")
    ang = math.degrees(math.atan2(y, x))
    if ang < 0:
        ang += 360.0
    return ang


# ---------------------- Normalização / validação ----------------------

def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    colmap: Dict[str, str] = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ["est", "estacao", "estação"]:
            colmap[c] = "EST"
        elif low in ["pv", "ponto visado", "ponto_visado", "ponto"]:
            colmap[c] = "PV"
        elif low in ["seq", "sequencia", "sequência", "serie", "série"]:
            colmap[c] = "SEQ"
        elif ("horizontal" in low and "pd" in low) or ("hz" in low and "pd" in low):
            colmap[c] = "Hz_PD"
        elif ("horizontal" in low and "pi" in low) or ("hz" in low and "pi" in low):
            colmap[c] = "Hz_PI"
        elif ("zenital" in low and "pd" in low) or ("z" in low and "pd" in low):
            colmap[c] = "Z_PD"
        elif ("zenital" in low and "pi" in low) or ("z" in low and "pi" in low):
            colmap[c] = "Z_PI"
        elif "dist" in low and "pd" in low:
            colmap[c] = "DI_PD"
        elif "dist" in low and "pi" in low:
            colmap[c] = "DI_PI"
        else:
            colmap[c] = c
    return df.rename(columns=colmap)


def validar_dataframe(df_original: pd.DataFrame):
    erros = []
    df = normalizar_colunas(df_original)

    missing = [c for c in REQUIRED_COLS_BASE if c not in df.columns]
    if missing:
        erros.append("Colunas obrigatórias ausentes: " + ", ".join(missing))

    for c in REQUIRED_COLS_ALL:
        if c not in df.columns:
            df[c] = ""

    invalid_rows_hz = []
    invalid_rows_z = []
    invalid_rows_di = []
    invalid_rows_seq = []

    for idx, row in df.iterrows():
        hz_pd = parse_angle_to_decimal(row.get("Hz_PD", ""))
        hz_pi = parse_angle_to_decimal(row.get("Hz_PI", ""))
        z_pd = parse_angle_to_decimal(row.get("Z_PD", ""))
        z_pi = parse_angle_to_decimal(row.get("Z_PI", ""))
        if np.isnan(hz_pd) or np.isnan(hz_pi):
            invalid_rows_hz.append(idx + 1)
        if np.isnan(z_pd) or np.isnan(z_pi):
            invalid_rows_z.append(idx + 1)
        try:
            di_pd = float(str(row.get("DI_PD", "")).replace(",", "."))
            di_pi = float(str(row.get("DI_PI", "")).replace(",", "."))
            if np.isnan(di_pd) or np.isnan(di_pi):
                invalid_rows_di.append(idx + 1)
        except Exception:
            invalid_rows_di.append(idx + 1)

        seq_val = str(row.get("SEQ", "")).strip()
        if seq_val != "":
            try:
                int(seq_val)
            except Exception:
                invalid_rows_seq.append(idx + 1)

    if invalid_rows_hz:
        erros.append(
            "Valores inválidos ou vazios em Hz_PD / Hz_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_hz))
        )
    if invalid_rows_z:
        erros.append(
            "Valores inválidos ou vazios em Z_PD / Z_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_z))
        )
    if invalid_rows_di:
        erros.append(
            "Valores inválidos ou vazios em DI_PD / DI_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_di))
        )
    if invalid_rows_seq:
        erros.append(
            "Valores inválidos em SEQ (devem ser inteiros) nas linhas: "
            + ", ".join(map(str, invalid_rows_seq))
        )

    if "SEQ" in df.columns:
        def _parse_seq(x):
            sx = str(x).strip()
            if sx == "":
                return np.nan
            try:
                return int(sx)
            except Exception:
                return np.nan

        df["SEQ"] = df["SEQ"].apply(_parse_seq)

    return df, erros


# --------------------------- Cálculo linha a linha ---------------------------

def calcular_linha_a_linha(df_uso: pd.DataFrame) -> pd.DataFrame:
    res = df_uso.copy()

    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    def calc_hz_medio(pd_deg, pi_deg):
        if math.isnan(pd_deg) or math.isnan(pi_deg):
            return float("nan")
        m = (pd_deg + pi_deg) / 2.0
        if pd_deg > pi_deg:
            hz = m + 90.0
        else:
            hz = m - 90.0
        return hz % 360.0

    res["Hz_med_deg"] = res.apply(
        lambda r: calc_hz_medio(r["Hz_PD_deg"], r["Hz_PI_deg"]), axis=1
    )
    res["Hz_med_DMS"] = res["Hz_med_deg"].apply(decimal_to_dms)

    def calc_z_corr(z_pd_deg, z_pi_deg):
        if math.isnan(z_pd_deg) or math.isnan(z_pi_deg):
            return float("nan")
        return (z_pd_deg - z_pi_deg) / 2.0 + 180.0

    res["Z_corr_deg"] = res.apply(
        lambda r: calc_z_corr(r["Z_PD_deg"], r["Z_PI_deg"]), axis=1
    )
    res["Z_corr_DMS"] = res["Z_corr_deg"].apply(decimal_to_dms)

    z_rad = res["Z_corr_deg"] * np.pi / 180.0
    res["DH_PD_m"] = np.abs(res["DI_PD_m"] * np.sin(z_rad)).round(3)
    res["DN_PD_m"] = np.abs(res["DI_PD_m"] * np.cos(z_rad)).round(3)
    res["DH_PI_m"] = np.abs(res["DI_PI_m"] * np.sin(z_rad)).round(3)
    res["DN_PI_m"] = np.abs(res["DI_PI_m"] * np.cos(z_rad)).round(3)

    res["DH_med_m"] = np.abs((res["DH_PD_m"] + res["DH_PI_m"]) / 2.0).round(3)
    res["DN_med_m"] = np.abs((res["DN_PD_m"] + res["DN_PI_m"]) / 2.0).round(3)

    return res


# --------------------------- Tabelas Hz / Z ---------------------------

def tabela_hz_por_serie(res: pd.DataFrame) -> pd.DataFrame:
    df = res.copy().reset_index(drop=False)
    df.rename(columns={"index": "_ordem_original"}, inplace=True)

    df["Hz_reduzido_deg"] = np.nan
    for est in df["EST"].unique():
        sub = df[df["EST"] == est]
        if sub.empty:
            continue
        ref = float(sub["Hz_med_deg"].min())
        mask = df["EST"] == est
        df.loc[mask, "Hz_reduzido_deg"] = (df.loc[mask, "Hz_med_deg"] - ref) % 360.0

    df["Hz_reduzido_DMS"] = df["Hz_reduzido_deg"].apply(decimal_to_dms)

    medias_series = []
    for (est, pv), sub in df.groupby(["EST", "PV"]):
        hz_list = [v for v in sub["Hz_reduzido_deg"].tolist() if not math.isnan(v)]
        hz_med_series = mean_direction_circular(hz_list)
        medias_series.append(
            {"EST": est, "PV": pv, "Hz_med_series_deg": hz_med_series}
        )
    df_med = pd.DataFrame(medias_series)
    df_med["Hz_med_series_DMS"] = df_med["Hz_med_series_deg"].apply(decimal_to_dms)

    df = df.merge(df_med, on=["EST", "PV"], how="left")
    df.sort_values(by="_ordem_original", inplace=True)

    tab = pd.DataFrame(
        {
            "Estação": df["EST"],
            "Ponto Visado": df["PV"],
            "Hz PD": df["Hz_PD"],
            "Hz PI": df["Hz_PI"],
            "Hz Médio": df["Hz_med_DMS"],
            "Hz Reduzido": df["Hz_reduzido_DMS"],
            "Média das séries": df["Hz_med_series_DMS"],
        }
    )
    return tab


def tabela_z_por_serie(res: pd.DataFrame) -> pd.DataFrame:
    df = res.copy().reset_index(drop=False)
    df.rename(columns={"index": "_ordem_original"}, inplace=True)

    medias_series = []
    for (est, pv), sub in df.groupby(["EST", "PV"]):
        z_vals = [v for v in sub["Z_corr_deg"].tolist() if not math.isnan(v)]
        if len(z_vals) == 0:
            z_med = float("nan")
        else:
            z_med = sum(z_vals) / len(z_vals)
        medias_series.append(
            {"EST": est, "PV": pv, "Z_med_series_deg": z_med}
        )
    df_med = pd.DataFrame(medias_series)
    df_med["Z_med_series_DMS"] = df_med["Z_med_series_deg"].apply(decimal_to_dms)

    df = df.merge(df_med, on=["EST", "PV"], how="left")
    df.sort_values(by="_ordem_original", inplace=True)

    tab = pd.DataFrame(
        {
            "Estação": df["EST"],
            "Ponto Visado": df["PV"],
            "Z PD": df["Z_PD"],
            "Z PI": df["Z_PI"],
            "Z Corrigido": df["Z_corr_DMS"],
            "Média das séries": df["Z_med_series_DMS"],
        }
    )
    return tab


# --------------------------- Resumo final ---------------------------

def tabela_resumo_final(res: pd.DataFrame, renomear_para_letras: bool = True) -> pd.DataFrame:
    tab_hz_full = tabela_hz_por_serie(res)
    tab_hz = (
        tab_hz_full
        .groupby(["Estação", "Ponto Visado"], as_index=False)
        .agg(
            **{
                "Hz Médio": ("Hz Médio", "first"),
                "Hz Reduzido": ("Hz Reduzido", "first"),
                "Média das séries": ("Média das séries", "first"),
            }
        )
    )

    tab_z_full = tabela_z_por_serie(res)
    tab_z = (
        tab_z_full
        .groupby(["Estação", "Ponto Visado"], as_index=False)
        .agg(
            **{
                "Z Corrigido": ("Z Corrigido", "first"),
                "Média Z das séries": ("Média das séries", "first"),
            }
        )
    )

    resumo = pd.merge(
        tab_hz,
        tab_z,
        on=["Estação", "Ponto Visado"],
        how="outer",
    )

    df_dh = res[["EST", "PV", "DH_med_m"]].copy()
    df_dh["DH_med_str"] = df_dh["DH_med_m"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else ""
    )
    df_dh_grp = df_dh.groupby(["EST", "PV"], as_index=False)["DH_med_str"].first()

    resumo = resumo.merge(
        df_dh_grp,
        left_on=["Estação", "Ponto Visado"],
        right_on=["EST", "PV"],
        how="left",
    )

    resumo = resumo[
        [
            "Estação",
            "Ponto Visado",
            "Hz Médio",
            "Hz Reduzido",
            "Média das séries",
            "Z Corrigido",
            "Média Z das séries",
            "DH_med_str",
        ]
    ].rename(
        columns={
            "Média das séries": "Média das séries (Hz)",
            "DH_med_str": "DH Médio (m)",
        }
    )

    if renomear_para_letras:
        mapa_simples = {"P1": "A", "P2": "B", "P3": "C"}
        resumo["EST"] = resumo["Estação"].astype(str).replace(mapa_simples)
        resumo["PV"] = resumo["Ponto Visado"].astype(str).replace(mapa_simples)
        resumo = resumo[
            [
                "EST",
                "PV",
                "Hz Médio",
                "Hz Reduzido",
                "Média das séries (Hz)",
                "Z Corrigido",
                "Média Z das séries",
                "DH Médio (m)",
            ]
        ]
    else:
        resumo = resumo[
            [
                "Estação",
                "Ponto Visado",
                "Hz Médio",
                "Hz Reduzido",
                "Média das séries (Hz)",
                "Z Corrigido",
                "Média Z das séries",
                "DH Médio (m)",
            ]
        ]
    return resumo


# --------------------------- Triângulo ---------------------------

def _angulo_interno(a: float, b: float, c: float) -> float:
    try:
        if a <= 0 or b <= 0 or c <= 0:
            return float("nan")
        cosA = (b**2 + c**2 - a**2) / (2 * b * c)
        cosA = max(-1.0, min(1.0, cosA))
        return math.degrees(math.acos(cosA))
    except Exception:
        return float("nan")


def _media_dh_entre_pontos(res: pd.DataFrame, pa: str, pb: str) -> float:
    vals = []
    for _, r in res.iterrows():
        e, v = str(r["EST"]), str(r["PV"])
        if {e, v} == {pa, pb}:
            vals.append(float(r["DH_med_m"]))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _direcao_media(res: pd.DataFrame, est: str, pv: str) -> float:
    vals = []
    for _, r in res.iterrows():
        if str(r["EST"]) == est and str(r["PV"]) == pv:
            vals.append(float(r["Hz_med_deg"]))
    if not vals:
        return float("nan")
    x = sum(math.cos(math.radians(a)) for a in vals)
    y = sum(math.sin(math.radians(a)) for a in vals)
    if x == 0 and y == 0:
        return float("nan")
    ang = math.degrees(math.atan2(y, x))
    if ang < 0:
        ang += 360
    return ang


def calcular_triangulo_duas_linhas(
    res: pd.DataFrame,
    idx1: int,
    idx2: int,
    estacao_op: str,
    conjunto_op: str,
) -> Optional[Dict]:
    """
    Usa duas linhas de 'res' para montar o triângulo.

    Caso geral:
      - estação = EST comum às duas linhas;
      - PV1 e PV2 são os pontos visados.

    Caso especial:
      - se estacao_op == 'A' e conjunto_op == '1ª leitura':
        estação geométrica forçada a P1; lados e direções P1–P2 e P1–P3.
    """
    if idx1 == idx2:
        return None
    if idx1 < 0 or idx1 >= len(res) or idx2 < 0 or idx2 >= len(res):
        return None

    r1 = res.iloc[idx1]
    r2 = res.iloc[idx2]

    est1, est2 = str(r1["EST"]), str(r2["EST"])
    pv1, pv2 = str(r1["PV"]), str(r2["PV"])

    if estacao_op == "A" and conjunto_op == "1ª leitura":
        est = "P1"

        AB = _media_dh_entre_pontos(res, "P1", "P2")
        AC = _media_dh_entre_pontos(res, "P1", "P3")
        if math.isnan(AB) or math.isnan(AC):
            return None

        hz12 = _direcao_media(res, "P1", "P2")
        hz13 = _direcao_media(res, "P1", "P3")
        if math.isnan(hz12) or math.isnan(hz13):
            return None

        ang_A_deg = (hz13 - hz12) % 360.0
        if ang_A_deg > 180.0:
            ang_A_deg = 360.0 - ang_A_deg

        BC = math.sqrt(
            AB**2 + AC**2 - 2 * AB * AC * math.cos(math.radians(ang_A_deg))
        )

        ang_B_deg = _angulo_interno(AC, AB, BC)  # em P2
        ang_C_deg = _angulo_interno(AB, AC, BC)  # em P3

        s = (AB + AC + BC) / 2.0
        area = math.sqrt(max(s * (s - AB) * (s - AC) * (s - BC), 0.0))

        info: Dict[str, object] = {
            "EST": "P1",
            "PV1": "P2",
            "PV2": "P3",
            "AB": AB,
            "AC": AC,
            "BC": BC,
            "ang_A_deg": ang_A_deg,
            "ang_B_deg": ang_B_deg,
            "ang_C_deg": ang_C_deg,
            "area_m2": area,
        }

    else:
        if est1 != est2 or pv1 == pv2:
            return None

        est = est1
        AB = float(r1["DH_med_m"])  # EST–PV1
        AC = float(r2["DH_med_m"])  # EST–PV2

        hz1 = float(r1["Hz_med_deg"])
        hz2 = float(r2["Hz_med_deg"])

        ang_A_deg = (hz2 - hz1) % 360.0
        if ang_A_deg > 180.0:
            ang_A_deg = 360.0 - ang_A_deg

        BC = math.sqrt(
            AB**2 + AC**2 - 2 * AB * AC * math.cos(math.radians(ang_A_deg))
        )

        ang_B_deg = _angulo_interno(AC, AB, BC)
        ang_C_deg = _angulo_interno(AB, AC, BC)

        s = (AB + AC + BC) / 2.0
        area = math.sqrt(max(s * (s - AB) * (s - AC) * (s - BC), 0.0))

        info = {
            "EST": est,
            "PV1": pv1,
            "PV2": pv2,
            "AB": AB,
            "AC": AC,
            "BC": BC,
            "ang_A_deg": ang_A_deg,
            "ang_B_deg": ang_B_deg,
            "ang_C_deg": ang_C_deg,
            "area_m2": area,
        }

    mapa_p_letra = {"P1": "A", "P2": "B", "P3": "C"}

    est = info["EST"]
    pv1 = info["PV1"]
    pv2 = info["PV2"]
    AB = info["AB"]
    AC = info["AC"]
    BC = info["BC"]

    lados_reais = [
        (est, pv1, AB),
        (est, pv2, AC),
        (pv1, pv2, BC),
    ]
    lados_rotulados = []
    for p_ini, p_fim, val in lados_reais:
        letra_ini = mapa_p_letra.get(p_ini, p_ini)
        letra_fim = mapa_p_letra.get(p_fim, p_fim)
        rot = f"{letra_ini}{letra_fim}"
        lados_rotulados.append((rot, p_ini, p_fim, val))

    angulos_reais = [
        (est, info["ang_A_deg"]),
        (pv1, info["ang_B_deg"]),
        (pv2, info["ang_C_deg"]),
    ]
    angulos_rotulados = []
    for p_nome, val in angulos_reais:
        letra = mapa_p_letra.get(p_nome, p_nome)
        angulos_rotulados.append((letra, p_nome, val))

    info["lados_ordenados"] = sorted(lados_rotulados, key=lambda x: x[3], reverse=True)
    info["angulos_ordenados"] = sorted(angulos_rotulados, key=lambda x: x[2], reverse=True)
    info["mapa_p_letra"] = mapa_p_letra

    return info


def selecionar_linhas_por_estacao_e_conjunto(
    res: pd.DataFrame, estacao_letra: str, conjunto: str
) -> Optional[Tuple[int, int]]:
    """
    Seleciona automaticamente duas linhas de 'res' para o triângulo, conforme
    estação (A,B,C) e conjunto (1ª,2ª,3ª).
    """
    letra_to_p = {"A": "P1", "B": "P2", "C": "P3"}
    est_ref = letra_to_p.get(estacao_letra)
    if est_ref is None:
        return None

    ordem_map = {"1ª leitura": 1, "2ª leitura": 2, "3ª leitura": 3}
    ordem = ordem_map.get(conjunto)
    if ordem is None:
        return None

    df = res.reset_index(drop=False).rename(columns={"index": "_idx_orig"})

    if est_ref == "P1":  # Estação A
        if ordem == 1:
            mask = (df["EST"] == "P2") & (df["PV"].isin(["P3", "P1"]))
        else:
            mask = (df["EST"] == "P1") & (df["PV"].isin(["P2", "P3"]))
    elif est_ref == "P2":  # Estação B
        mask = (df["EST"] == "P2") & (df["PV"].isin(["P3", "P1"]))
    else:  # P3, Estação C
        mask = (df["EST"] == "P3") & (df["PV"].isin(["P1", "P2"]))

    cand = df[mask].sort_values(by="_idx_orig")
    if len(cand) < 2:
        return None

    cand = cand.reset_index(drop=True)
    cand["par_id"] = cand.index // 2

    par_desejado = ordem - 1
    par = cand[cand["par_id"] == par_desejado]
    if len(par) < 2:
        return None

    idxs = par["_idx_orig"].tolist()[:2]
    return int(idxs[0]), int(idxs[1])


def gerar_modelo_excel_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_id = pd.DataFrame(
            {
                "Campo": [
                    "Professor(a)",
                    "Equipamento",
                    "Dados",
                    "Local",
                    "Patrimônio",
                ],
                "Valor": ["", "", "", "", ""],
            }
        )
        df_id.to_excel(writer, sheet_name="Identificação", index=False)

        df_dados = pd.DataFrame(
            {
                "EST": ["P1", "P1", "P1", "P1"],
                "PV": ["P2", "P3", "P2", "P3"],
                "SEQ": [1, 1, 2, 2],
                "Hz_PD": ["00°00'00\"", "18°58'22\"", "00°01'01\"", "18°59'34\""],
                "Hz_PI": ["179°59'48\"", "198°58'14\"", "180°00'45\"", "198°59'24\""],
                "Z_PD": ["90°51'08\"", "90°51'25\"", "90°51'06\"", "90°51'24\""],
                "Z_PI": ["269°08'52\"", "269°08'33\"", "269°08'50\"", "269°08'26\""],
                "DI_PD": [25.365, 26.285, 25.365, 26.285],
                "DI_PI": [25.365, 26.285, 25.365, 26.285],
            }
        )
        df_dados.to_excel(writer, sheet_name="Dados", index=False)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# 3. PLOTAGEM E XLSX – (equivalente ao plotting.py)
# =============================================================================

def plotar_triangulo_info(info: Dict, estacao_op: str, conjunto_op: str):
    """
    Desenha o triângulo em planta.

    - O vértice em (0,0) é info["EST"] (P1, P2 ou P3).
    - Os outros vértices são PV1 e PV2.
    """
    est = info["EST"]
    pv1 = info["PV1"]
    pv2 = info["PV2"]

    AB = info["AB"]
    AC = info["AC"]
    BC = info["BC"]

    x_E, y_E = 0.0, 0.0
    x_V2, y_V2 = AC, 0.0

    if AC == 0:
        x_V1, y_V1 = AB, 0.0
    else:
        x_V1 = (AB**2 - BC**2 + AC**2) / (2 * AC)
        arg = max(AB**2 - x_V1**2, 0.0)
        y_V1 = math.sqrt(arg)

    xs = [x_E, x_V1, x_V2, x_E]
    ys = [y_E, y_V1, y_V2, y_E]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, "-o", color="#7f0000")
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.set_aspect("equal", "box")

    ax.text(x_E, y_E, f"{est}", fontsize=10, color="#111827")
    ax.text(x_V1, y_V1, f"{pv1}", fontsize=10, color="#111827")
    ax.text(x_V2, y_V2, f"{pv2}", fontsize=10, color="#111827")

    ax.text(
        (x_E + x_V1) / 2,
        (y_E + y_V1) / 2,
        f"{est}–{pv1} = {AB:.3f} m",
        color="#374151",
        fontsize=9,
    )
    ax.text(
        (x_E + x_V2) / 2,
        (y_E + y_V2) / 2,
        f"{est}–{pv2} = {AC:.3f} m",
        color="#374151",
        fontsize=9,
    )
    ax.text(
        (x_V1 + x_V2) / 2,
        (y_V1 + y_V2) / 2,
        f"{pv1}–{pv2} = {BC:.3f} m",
        color="#374151",
        fontsize=9,
    )

    ax.set_xlabel("X (m)", color="#111827")
    ax.set_ylabel("Y (m)", color="#111827")
    ax.tick_params(colors="#111827")
    ax.grid(True, linestyle="--", alpha=0.3, color="#9ca3af")
    ax.set_title(
        "Representação do triângulo em planta (ponto de vista na estação)",
        color="#111827",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf, fig


def gerar_xlsx_com_figura(info_triangulo: Dict, figura_buf: io.BytesIO) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        df_resumo = pd.DataFrame(
            {
                "Descrição": [
                    "Lado estação–PV1",
                    "Lado estação–PV2",
                    "Lado PV1–PV2",
                    "Ângulo interno na estação",
                    "Ângulo interno no PV1",
                    "Ângulo interno no PV2",
                    "Área do triângulo (m²)",
                ],
                "Valor": [
                    f"{info_triangulo['AB']:.3f} m",
                    f"{info_triangulo['AC']:.3f} m",
                    f"{info_triangulo['BC']:.3f} m",
                    decimal_to_dms(info_triangulo["ang_A_deg"]),
                    decimal_to_dms(info_triangulo["ang_B_deg"]),
                    decimal_to_dms(info_triangulo["ang_C_deg"]),
                    f"{info_triangulo['area_m2']:.3f}",
                ],
            }
        )
        df_resumo.to_excel(writer, sheet_name="ResumoTriangulo", index=False)

        ws_fig = wb.add_worksheet("FiguraTriangulo")
        writer.sheets["FiguraTriangulo"] = ws_fig
        if figura_buf is not None:
            ws_fig.insert_image("B2", "triangulo.jpg", {"image_data": figura_buf})

    output.seek(0)
    return output.getvalue()


# =============================================================================
# 4. INTERFACE STREAMLIT (tudo em um arquivo)
# =============================================================================

st.set_page_config(
    page_title="Calculadora de Ângulos e Distâncias | UFPE",
    layout="wide",
    page_icon="📐",
)

# --------------------------- CSS ---------------------------

CUSTOM_CSS = """
<style>
body, .stApp {
  background:#f3f4f6;
  color: #111827;
  font-family:"Trebuchet MS",system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
}

.main-card{
  background:#ffffff;
  color: #111827;
  border-radius:22px;
  padding:1.4rem 2.0rem 1.4rem 2.0rem;
  border:1px solid rgba(148,27,37,0.20);
  box-shadow:0 18px 40px rgba(15,23,42,0.18);
  max-width:1320px;
  margin:1.2rem auto 2.0rem auto;
}
.main-card p { text-align: justify; }

.ufpe-header-band{
  width: 100%;
  padding:0.7rem 1.0rem 0.6rem 1.0rem;
  border-radius:14px;
  background:linear-gradient(90deg,#4b0000 0%,#7e0000 40%,#b30000 75%,#4b0000 100%);
  color:#f9fafb;
  display:flex;
  align-items:flex-start;
  gap:0.8rem;
}
.ufpe-header-text{
  font-size:0.87rem;
}
.ufpe-header-text b{
  font-weight:700;
}

.section-title{
  font-size:1.00rem;
  font-weight:700;
  margin-top:1.5rem;
  margin-bottom:0.6rem;
  display:flex;
  align-items:center;
  gap:0.4rem;
  color:#8b0000;
  text-transform:uppercase;
  letter-spacing:0.05em;
}
.section-title span.dot{
  width:9px;
  height:9px;
  border-radius:999px;
  background:radial-gradient(circle at 30% 30%,#ffffff 0%,#ffbdbd 35%,#7f0000 90%);
}

.helper-box{
  border-radius:10px;
  padding:0.6rem 0.8rem;
  background:#fff5f5;
  border:1px solid rgba(148,27,37,0.35);
  font-size:0.86rem;
  color:#111827;
  margin-bottom:0.5rem;
}

[data-testid="stDataFrame"],[data-testid="stDataEditor"]{
  background:#ffffff !important;
  border-radius:10px;
  border:1px solid rgba(148,27,37,0.25);
  box-shadow:0 10px 22px rgba(15,23,42,0.12);
}

.stButton>button, .stDownloadButton>button {
  background: #b30000;
  color: #ffffff;
  border-radius: 999px;
  border: 1px solid #7f0000;
  padding: 0.35rem 1.1rem;
  font-weight: 600;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background: #ffffff;
  color: #111827;
  border: 1px solid #b30000;
}

.footer-text{
  font-size:0.75rem;
  color:#6b7280;
}

:root{color-scheme:light;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------- Cabeçalho UFPE ---------------------------

def cabecalho_ufpe(info_id):
    prof = info_id.get("Professor(a)", "")
    equip = info_id.get("Equipamento", "")
    data_str = info_id.get("Dados", "")
    local = info_id.get("Local", "")
    patr = info_id.get("Patrimônio", "")

    def linha(label, value):
        if value:
            return f"{label}: <u>{value}</u><br>"
        else:
            return f"{label}: _________________________________<br>"

    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.markdown("<div class='ufpe-header-band'>", unsafe_allow_html=True)
    col_logo, col_text = st.columns([1, 9])
    with col_logo:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/8/85/Bras%C3%A3o_da_UFPE.png",
            width=70,
        )
    with col_text:
        texto = (
            "<div class='ufpe-header-text'>"
            "<b>UNIVERSIDADE FEDERAL DE PERNAMBUCO - UFPE</b><br>"
            "DECART — Departamento de Engenharia Cartográfica<br>"
            "LATOP — Laboratório de Topografia<br>"
            "Curso: Engenharia Cartográfica e Agrimensura<br>"
            "Disciplina: Equipamentos de Medição<br>"
            f"{linha('Professor(a)', prof)}"
            f"{linha('Equipamento', equip)}"
            f"{linha('Data', data_str)}"
            f"{linha('Local', local)}"
            f"{linha('Patrimônio', patr)}"
            "</div>"
        )
        st.markdown(texto, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style="margin-top:0.9rem;font-size:1.5rem;font-weight:800;color:#7f0000;">
            Calculadora de Ângulos e Distâncias – Método das Direções para Triângulos
        </p>
        <p style="font-size:0.92rem;">
            Esta ferramenta auxilia no processamento das leituras obtidas com estação total,
            calculando médias de direções horizontais (Hz), ângulos verticais/zenitais,
            distâncias horizontais médias e a geometria do triângulo formado pelos
            pontos P1, P2 e P3.
        </p>
        <div class="helper-box">
            <b>Preenchimento dos dados de identificação:</b><br>
            Os campos Professor(a), Equipamento, Dados, Local e Patrimônio são lidos
            automaticamente da aba <b>Identificação</b> do modelo Excel. Os dados são exibidos
            no formato <b>DD/MM/AAAA</b>. Caso algum campo venha em branco, ele pode ser
            completado manualmente no arquivo exportado.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------- Página 1 – modelo + upload ---------------------------

def pagina_carregar_dados():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>1. Modelo de planilha</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    modelo_bytes = gerar_modelo_excel_bytes()
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=modelo_bytes,
        file_name="modelo_medicao_direcoes_ufpe.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.markdown(
        """
        <p style="font-size:0.9rem;">
        O modelo contém duas abas: <b>Identificação</b> (dados do cabeçalho)
        e <b>Dados</b> (leituras Hz, Z e distâncias).
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>2. Carregar dados de campo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Envie o arquivo Excel (com abas Identificação e Dados)",
        type=["xlsx", "xls"],
    )

    if uploaded is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        xls = pd.ExcelFile(uploaded)

        # Identificação
        sheet_id = None
        for s in xls.sheet_names:
            if s.strip().lower() in ["identificação", "identificacao"]:
                sheet_id = s
                break
        info_id = {
            "Professor(a)": "",
            "Equipamento": "",
            "Dados": "",
            "Local": "",
            "Patrimônio": "",
        }
        if sheet_id is not None:
            df_id = pd.read_excel(xls, sheet_name=sheet_id)
            info_id = ler_identificacao_from_df(df_id)

        # Dados
        sheet_dados = None
        for s in xls.sheet_names:
            if s.strip().lower() in ["dados", "medicoes", "medições"]:
                sheet_dados = s
                break
        if sheet_dados is None:
            sheet_dados = xls.sheet_names[0]

        raw_df = pd.read_excel(xls, sheet_name=sheet_dados)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.success(
        f"Arquivo '{uploaded.name}' carregado. Aba de dados utilizada: '{sheet_dados}'."
    )

    df_valid, erros = validar_dataframe(raw_df)

    st.subheader("Pré-visualização dos dados importados")
    cols_to_show = [c for c in REQUIRED_COLS_ALL if c in df_valid.columns]
    st.dataframe(df_valid[cols_to_show], use_container_width=True)

    if erros:
        st.error("Não foi possível calcular devido aos seguintes problemas:")
        for e in erros:
            st.markdown(f"- {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cols_use = [c for c in REQUIRED_COLS_ALL if c in df_valid.columns]
    df_uso = df_valid[cols_use].copy()

    st.session_state["df_uso"] = df_uso
    st.session_state["info_id"] = info_id

    if st.button("Ir para processamento"):
        st.session_state["pagina"] = "processamento"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------- Página 2 – processamento ---------------------------

def pagina_processamento():
    if "df_uso" not in st.session_state or "info_id" not in st.session_state:
        st.warning("Nenhum dado carregado. Volte à página 'Carregar dados' primeiro.")
        if st.button("Voltar para carregar dados"):
            st.session_state["pagina"] = "carregar"
            st.rerun()
        return

    df_uso = st.session_state["df_uso"]
    info_id = st.session_state["info_id"]

    cabecalho_ufpe(info_id)

    st_local = st

    # 3. Linha a linha
    st_local.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>3. Cálculo de Hz, Z e distâncias (linha a linha)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    res = calcular_linha_a_linha(df_uso)

    cols_linha = [
        "EST",
        "PV",
        "SEQ",
        "Hz_PD",
        "Hz_PI",
        "Hz_med_DMS",
        "Z_PD",
        "Z_PI",
        "Z_corr_DMS",
        "DH_PD_m",
        "DH_PI_m",
        "DH_med_m",
    ]
    df_linha = res[cols_linha].copy()
    for c in ["DH_PD_m", "DH_PI_m", "DH_med_m"]:
        df_linha[c] = df_linha[c].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )
    st_local.dataframe(df_linha, use_container_width=True)

    # 4. Hz
    st_local.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Medição Angular Horizontal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_hz = tabela_hz_por_serie(res)
    st_local.dataframe(tab_hz, use_container_width=True)

    # 5. Z
    st_local.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>5. Medição Angular Vertical / Zenital</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_z = tabela_z_por_serie(res)
    st_local.dataframe(tab_z, use_container_width=True)

    # 6. Resumo
    st_local.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>6. Tabela resumo (Hz, Z e DH)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    resumo = tabela_resumo_final(res, renomear_para_letras=True)
    st_local.dataframe(resumo, use_container_width=True)

    # 7. Triângulo
    st_local.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>7. TRIÂNGULO SELECIONADO (CONJUNTO AUTOMÁTICO DE MEDIÇÕES)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st_local.columns(2)
    with col_a:
        estacao_op = st_local.selectbox("Estação (A, B, C)", ["A", "B", "C"])
    with col_b:
        conjunto_op = st_local.selectbox(
            "Conjunto de leituras",
            ["1ª leitura", "2ª leitura", "3ª leitura"],
        )

    st_local.markdown(
        "<p>O programa seleciona automaticamente o par de leituras adequadas "
        "para formar o triângulo, conforme as regras definidas para cada estação.</p>",
        unsafe_allow_html=True,
    )

    if st_local.button("Gerar triângulo"):
        pares = selecionar_linhas_por_estacao_e_conjunto(res, estacao_op, conjunto_op)
        if pares is None:
            st_local.error(
                "Não foi possível encontrar duas leituras compatíveis para "
                f"Estação {estacao_op} e {conjunto_op}. "
                "Verifique se a ordem das linhas (EST, PV) segue o modelo."
            )
        else:
            idx1, idx2 = pares
            info = calcular_triangulo_duas_linhas(res, idx1, idx2, estacao_op, conjunto_op)
            if info is None:
                st_local.error(
                    "Falha ao calcular o triângulo a partir das leituras selecionadas."
                )
            else:
                est = info["EST"]
                pv1 = info["PV1"]
                pv2 = info["PV2"]

                st_local.markdown(
                    f"<p><b>Triângulo formado automaticamente pelos pontos {est}, {pv1} e {pv2} "
                    f"(conjunto: {conjunto_op}, estação selecionada: {estacao_op}).</b></p>",
                    unsafe_allow_html=True,
                )

                lados_ord = info.get("lados_ordenados", [])
                ang_ord = info.get("angulos_ordenados", [])

                col1, col2 = st_local.columns(2)
                with col1:
                    st_local.markdown("**Lados (m) – do maior para o menor:**")
                    linhas_lados = []
                    for rot, p_ini, p_fim, val in lados_ord:
                        linhas_lados.append(
                            f"- {p_ini}–{p_fim} ({rot}): ` {val:.3f} ` m"
                        )
                    st_local.markdown("\n".join(linhas_lados))

                    st_local.markdown("**Ângulos internos – do maior para o menor:**")
                    linhas_ang = []
                    for letra, p_nome, val in ang_ord:
                        linhas_ang.append(
                            f"- Em {p_nome} ({letra}): ` {decimal_to_dms(val)} `"
                        )
                    st_local.markdown("\n".join(linhas_ang))

                    st_local.markdown(
                        f"**Área do triângulo:** ` {info['area_m2']:.3f} ` m²"
                    )

                with col2:
                    img_buf, fig = plotar_triangulo_info(info, estacao_op, conjunto_op)
                    st_local.pyplot(fig)

                xlsx_bytes = gerar_xlsx_com_figura(info, img_buf)
                st_local.download_button(
                    "📊 Baixar XLSX com resumo e figura do triângulo",
                    data=xlsx_bytes,
                    file_name="triangulo_ufpe_resumo_figura.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    st_local.markdown(
        """
        <p class="footer-text">
            Versão (arquivo único): <code>UFPE_v21_single — triângulo com ponto de vista na estação real (P1/P2/P3),
            croqui sem letras A/B/C nos vértices, caso especial Estação A / 1ª leitura com P1 na base esquerda (0,0),
            cabeçalho com data em formato DD/MM/AAAA.</code>
        </p>
        """,
        unsafe_allow_html=True,
    )


# --------------------------- Router de “páginas” ---------------------------

if "pagina" not in st.session_state:
    st.session_state["pagina"] = "carregar"

if st.session_state["pagina"] == "carregar":
    pagina_carregar_dados()
else:
    pagina_processamento()
