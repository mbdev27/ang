# app.py
# UFPE - Calculadora de Ângulos e Distâncias
# Versão: Triângulo por série usando apenas DH da série + Teorema dos Cossenos

import io
import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==================== Config página ====================
st.set_page_config(
    page_title="Calculadora de Ângulos e Distâncias | UFPE",
    layout="wide",
    page_icon="📐",
)

# ==================== Parâmetros globais ====================

REQUIRED_COLS = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]

# Mapa clássico P1–P3 para Ré/Vante
RE_VANTE_MAP: Dict[str, Tuple[str, str]] = {
    "P1": ("P2", "P3"),
    "P2": ("P1", "P3"),
    "P3": ("P1", "P2"),
}

# ==================== Funções auxiliares de ângulo ====================

def parse_angle_to_decimal(value: str) -> float:
    """Converte ângulos em string (DMS ou decimal) para graus decimais."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "":
        return float("nan")
    # caso já seja decimal
    try:
        if all(ch.isdigit() or ch in ".,-+" for ch in s):
            return float(s.replace(",", "."))
    except Exception:
        pass
    # limpar símbolos
    for ch in ["°", "º", "'", "’", "´", "′", '"', "″"]:
        s = s.replace(ch, " ")
    s = s.replace(",", ".")
    parts = [p for p in s.split() if p != ""]
    if len(parts) == 0:
        return float("nan")
    try:
        deg = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
    except Exception:
        return float("nan")
    sign = 1.0
    if deg < 0:
        sign = -1.0
        deg = abs(deg)
    return sign * (deg + minutes / 60.0 + seconds / 3600.0)


def decimal_to_dms(angle_deg: float) -> str:
    """Converte graus decimais para string DMS."""
    if angle_deg is None or math.isnan(angle_deg):
        return ""
    sign = "-" if angle_deg < 0 else ""
    a = abs(angle_deg)
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
    return f"{sign}{d:02d}°{m:02d}'{s:02d}\""


def mean_direction_two(a_deg: float, b_deg: float) -> float:
    """Média vetorial de duas direções (graus)."""
    if math.isnan(a_deg) or math.isnan(b_deg):
        return float("nan")
    a_rad = math.radians(a_deg)
    b_rad = math.radians(b_deg)
    x = math.cos(a_rad) + math.cos(b_rad)
    y = math.sin(a_rad) + math.sin(b_rad)
    if x == 0 and y == 0:
        return float("nan")
    ang = math.degrees(math.atan2(y, x))
    if ang < 0:
        ang += 360.0
    return ang


def mean_direction_list(angles_deg: pd.Series) -> float:
    """Média vetorial de uma lista de direções (graus)."""
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


# ==================== Normalização / Validação ====================

def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para EST, PV, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI."""
    df = df_original.copy()
    colmap = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ["est", "estacao", "estação"]:
            colmap[c] = "EST"
        elif low in ["pv", "ponto visado", "ponto_visado", "ponto"]:
            colmap[c] = "PV"
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


def validar_dataframe(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Valida presença de colunas e valores numéricos/angulares básicos."""
    erros: List[str] = []
    df = normalizar_colunas(df_original)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        erros.append("Colunas obrigatórias ausentes: " + ", ".join(missing))

    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""

    invalid_rows_hz = []
    invalid_rows_z = []
    invalid_rows_di = []

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

    if invalid_rows_hz:
        erros.append(
            "Valores inválidos ou vazios em Hz_PD / Hz_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_hz))
            + "."
        )
    if invalid_rows_z:
        erros.append(
            "Valores inválidos ou vazios em Z_PD / Z_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_z))
            + "."
        )
    if invalid_rows_di:
        erros.append(
            "Valores inválidos ou vazios em DI_PD / DI_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_di))
            + "."
        )

    return df, erros


# ==================== Cálculos linha a linha / por par ====================

def calcular_linha_a_linha(df_uso: pd.DataFrame) -> pd.DataFrame:
    """Calcula Hz, Z, DH, DN e médias (PD/PI) linha a linha."""
    res = df_uso.copy()
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    z_pd_rad = res["Z_PD_deg"] * np.pi / 180.0
    z_pi_rad = res["Z_PI_deg"] * np.pi / 180.0

    res["DH_PD_m"] = np.abs(res["DI_PD_m"] * np.sin(z_pd_rad)).round(3)
    res["DN_PD_m"] = np.abs(res["DI_PD_m"] * np.cos(z_pd_rad)).round(3)
    res["DH_PI_m"] = np.abs(res["DI_PI_m"] * np.sin(z_pi_rad)).round(3)
    res["DN_PI_m"] = np.abs(res["DI_PI_m"] * np.cos(z_pi_rad)).round(3)

    res["Hz_med_deg"] = res.apply(
        lambda r: mean_direction_two(r["Hz_PD_deg"], r["Hz_PI_deg"]), axis=1
    )
    res["Hz_med_DMS"] = res["Hz_med_deg"].apply(decimal_to_dms)

    res["DH_med_m"] = np.abs((res["DH_PD_m"] + res["DH_PI_m"]) / 2.0).round(3)
    res["DN_med_m"] = np.abs((res["DN_PD_m"] + res["DN_PI_m"]) / 2.0).round(3)

    return res


def agregar_por_par(res: pd.DataFrame) -> pd.DataFrame:
    """Agrega as séries por par EST–PV, gerando médias por par."""
    def agg_par(df_group: pd.DataFrame) -> pd.Series:
        out = {}
        out["Hz_PD_med_deg"] = mean_direction_list(df_group["Hz_PD_deg"])
        out["Hz_PI_med_deg"] = mean_direction_list(df_group["Hz_PI_deg"])
        out["Z_PD_med_deg"] = mean_direction_list(df_group["Z_PD_deg"])
        out["Z_PI_med_deg"] = mean_direction_list(df_group["Z_PI_deg"])
        out["DI_PD_med_m"] = float(df_group["DI_PD_m"].mean())
        out["DI_PI_med_m"] = float(df_group["DI_PI_m"].mean())
        out["DH_med_m_par"] = float(df_group["DH_med_m"].mean())
        return pd.Series(out)

    df_par = res.groupby(["EST", "PV"], as_index=False).apply(agg_par)

    df_par["Hz_med_deg_par"] = df_par.apply(
        lambda r: mean_direction_two(r["Hz_PD_med_deg"], r["Hz_PI_med_deg"]), axis=1
    )
    df_par["Hz_med_DMS_par"] = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    return df_par


# ==================== Tabela Hz e Z ====================

def construir_tabela_hz_com_re_vante(df_par: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela de médias Hz por par e tabela Hz Ré/Vante por estação."""
    hz_pd_med_dms = df_par["Hz_PD_med_deg"].apply(decimal_to_dms)
    hz_pi_med_dms = df_par["Hz_PI_med_deg"].apply(decimal_to_dms)
    hz_med_dms = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    base = pd.DataFrame(
        {
            "EST": df_par["EST"],
            "PV": df_par["PV"],
            "Hz PD (médio)": hz_pd_med_dms,
            "Hz PI (médio)": hz_pi_med_dms,
            "Hz Médio (PD/PI)": hz_med_dms,
            "Hz_med_deg_par": df_par["Hz_med_deg_par"],
        }
    )

    rows_re_vante = []
    for est, (pv_re, pv_vante) in RE_VANTE_MAP.items():
        sub_est = base[base["EST"] == est].copy()
        if sub_est.empty:
            continue
        hz_re_s = sub_est.loc[sub_est["PV"] == pv_re, "Hz_med_deg_par"]
        hz_va_s = sub_est.loc[sub_est["PV"] == pv_vante, "Hz_med_deg_par"]
        if len(hz_re_s) == 0 or len(hz_va_s) == 0:
            continue
        hz_re = hz_re_s.iloc[0]
        hz_va = hz_va_s.iloc[0]
        alpha = (hz_va - hz_re + 360.0) % 360.0
        rows_re_vante.append(
            {
                "EST": est,
                "PV_Ré": pv_re,
                "PV_Vante": pv_vante,
                "Hz_Ré (deg)": hz_re,
                "Hz_Vante (deg)": hz_va,
                "Hz_Ré (DMS)": decimal_to_dms(hz_re),
                "Hz_Vante (DMS)": decimal_to_dms(hz_va),
                "α (deg)": alpha,
                "α (DMS)": decimal_to_dms(alpha),
            }
        )
    df_hz_re_vante = pd.DataFrame(rows_re_vante)
    return base, df_hz_re_vante


def tabela_medicao_angular_vertical(df_par: pd.DataFrame) -> pd.DataFrame:
    """Tabela de medição angular vertical/zenital média."""
    z_pd_med = df_par["Z_PD_med_deg"]
    z_pi_med = df_par["Z_PI_med_deg"]
    z_corr_deg = (z_pd_med - z_pi_med) / 2.0 + 180.0
    z_pd_med_dms = z_pd_med.apply(decimal_to_dms)
    z_pi_med_dms = z_pi_med.apply(decimal_to_dms)
    z_corr_dms = z_corr_deg.apply(decimal_to_dms)
    tab = pd.DataFrame(
        {
            "EST": df_par["EST"],
            "PV": df_par["PV"],
            "Z PD (médio)": z_pd_med_dms,
            "Z PI (médio)": z_pi_med_dms,
            "Z Corrigido": z_corr_dms,
            "Média das Séries": z_corr_dms,
        }
    )
    return tab


# ==================== Distâncias médias simétricas (para diagnóstico) ====================

def tabela_distancias_medias_simetricas(df_par: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de df_par (com DH_med_m_par por EST,PV),
    monta uma tabela de distâncias médias simétricas por par de pontos:
    chave = {A,B}, valor = média das DH de A->B e B->A (se existirem).
    """
    aux = df_par[["EST", "PV", "DH_med_m_par"]].copy()
    registros: Dict[Tuple[str, str], List[float]] = {}

    for _, row in aux.iterrows():
        a = str(row["EST"])
        b = str(row["PV"])
        if a == b:
            continue
        par = tuple(sorted([a, b]))
        dh = float(row["DH_med_m_par"])
        if par not in registros:
            registros[par] = []
        registros[par].append(dh)

    linhas = []
    for (a, b), valores in registros.items():
        dh_med = float(np.mean(valores))
        linhas.append({"PontoA": a, "PontoB": b, "DH_media": dh_med})

    df_dist = pd.DataFrame(linhas)
    if not df_dist.empty:
        df_dist.sort_values("DH_media", ascending=False, inplace=True)
    return df_dist


# ==================== Azimute de referência / coordenadas médias (opcional) ====================

def delta_from_azimuth(az_deg: float, dh: float) -> Tuple[float, float]:
    """Retorna (ΔE, ΔN) a partir de azimute e distância horizontal."""
    az_rad = math.radians(az_deg)
    de = dh * math.sin(az_rad)
    dn = dh * math.cos(az_rad)
    return de, dn


def calcular_azimutes_corrigidos(df_par: pd.DataFrame, az_ref: float, est_inicio: str, est_segundo: str) -> pd.DataFrame:
    """Aplica azimute de referência ao par est_inicio→est_segundo e corrige todos os azimutes médios."""
    df_par = df_par.copy()
    mask_ref = (df_par["EST"] == est_inicio) & (df_par["PV"] == est_segundo)
    if not mask_ref.any():
        df_par["Az_corrigido_deg"] = df_par["Hz_med_deg_par"] % 360.0
        df_par["Az_corrigido_DMS"] = df_par["Az_corrigido_deg"].apply(decimal_to_dms)
        return df_par
    hz_ref = df_par.loc[mask_ref, "Hz_med_deg_par"].iloc[0]
    offset = az_ref - hz_ref
    df_par["Az_corrigido_deg"] = (df_par["Hz_med_deg_par"] + offset) % 360.0
    df_par["Az_corrigido_DMS"] = df_par["Az_corrigido_deg"].apply(decimal_to_dms)
    return df_par


def calcular_coordenadas(df_par_az: pd.DataFrame, est_inicio: str) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Calcula coordenadas aproximadas (E,N) a partir de azimutes e DH médios por par."""
    coords: Dict[str, Tuple[float, float]] = {}
    coords[est_inicio] = (0.0, 0.0)

    aux_rows = []
    for _, r in df_par_az.iterrows():
        est = str(r["EST"])
        pv = str(r["PV"])
        az = r["Az_corrigido_deg"]
        dh = r.get("DH_med_m_par", np.nan)
        if math.isnan(az) or math.isnan(dh):
            continue
        aux_rows.append({"EST": est, "PV": pv, "az_deg": az, "Dh_m": dh})
    aux_df = pd.DataFrame(aux_rows)

    max_iters = 20
    for _ in range(max_iters):
        changed = False
        for _, row in aux_df.iterrows():
            est = row["EST"]
            pv = row["PV"]
            az = row["az_deg"]
            dh = row["Dh_m"]
            if est in coords and pv not in coords:
                de, dn = delta_from_azimuth(az, dh)
                e0, n0 = coords[est]
                coords[pv] = (e0 + de, n0 + dn)
                changed = True
            elif pv in coords and est not in coords:
                de, dn = delta_from_azimuth(az, dh)
                e1, n1 = coords[pv]
                coords[est] = (e1 - de, n1 - dn)
                changed = True
        if not changed:
            break

    rows = []
    for pt, (e, n) in coords.items():
        rows.append({"Ponto": pt, "E (m)": round(e, 3), "N (m)": round(n, 3)})
    return pd.DataFrame(rows), coords


# ==================== Triângulo por série (apenas DH + cossenos) ====================

def detectar_tres_pontos(df_par: pd.DataFrame) -> List[str]:
    """
    Retorna a lista de pontos distintos (EST e PV).
    Se houver exatamente 3, trataremos como triângulo.
    """
    pontos = set(df_par["EST"]).union(set(df_par["PV"]))
    return sorted(pontos)


def numerar_series_por_estacao(res_linha: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna SERIE = índice da série por estação."""
    df = res_linha.copy()
    df["SERIE"] = df.groupby("EST").cumcount().astype(int) + 1
    return df


def obter_dh_triangulo_por_serie(res_serie: pd.DataFrame, pontos: List[str]):
    """
    pontos: [A, B, C] (rótulos, ex.: ["P1","P2","P3"])
    res_serie: dataframe linha a linha, com EST, PV, SERIE, DH_med_m.

    Retorna:
        { serie: { 'AB': dh_AB, 'BC': dh_BC, 'AC': dh_AC } }
    usando DH_med_m da série (não a média final por par).
    """
    if len(pontos) != 3:
        return {}

    A, B, C = pontos
    resultados: Dict[int, Dict[str, float]] = {}

    # número de séries por estação (pegamos o mínimo)
    n_series = int(res_serie.groupby("EST")["SERIE"].max().min())

    for s in range(1, n_series + 1):
        def dh_serie(est, pv):
            linha = res_serie[
                (res_serie["EST"] == est)
                & (res_serie["PV"] == pv)
                & (res_serie["SERIE"] == s)
            ]
            if linha.empty:
                return None
            return float(linha["DH_med_m"].iloc[0])

        # aqui assumimos leituras A->B, B->C, A->C na planilha
        dh_AB = dh_serie(A, B)
        dh_BC = dh_serie(B, C)
        dh_AC = dh_serie(A, C)

        if dh_AB is None or dh_BC is None or dh_AC is None:
            continue

        resultados[s] = {"AB": dh_AB, "BC": dh_BC, "AC": dh_AC}

    return resultados


def montar_triangulo_dh(DH_AB: float, DH_BC: float, DH_AC: float):
    """
    Recebe as três distâncias (DH) do triângulo: AB, BC, AC.
    Monta coordenadas artificiais (x,y) para A,B,C usando:
        - teorema dos cossenos,
        - base no lado AC.
    Retorna:
        coords: {"A":(xA,yA), "B":(xB,yB), "C":(xC,yC)}
        angulos: {"A":α_A_deg, "B":α_B_deg, "C":α_C_deg}
        area: área (m²)
    """
    L_AB = DH_AB
    L_BC = DH_BC
    L_AC = DH_AC

    # 1) Define A e C na base do eixo X
    A = (0.0, 0.0)
    C = (L_AC, 0.0)

    # 2) Ângulo em A, oposto a BC (lado a = BC), pelo teorema dos cossenos
    a = L_BC
    b = L_AC
    c = L_AB
    cos_A = (b*b + c*c - a*a) / (2.0 * b * c)
    cos_A = max(min(cos_A, 1.0), -1.0)  # evitar erro numérico
    ang_A_rad = math.acos(cos_A)
    ang_A = math.degrees(ang_A_rad)

    # 3) Coordenadas de B a partir de A
    Bx = L_AB * math.cos(ang_A_rad)
    By = L_AB * math.sin(ang_A_rad)
    B = (Bx, By)

    # 4) Ângulo em B, oposto a AC
    cos_B = (L_AB*L_AB + L_BC*L_BC - L_AC*L_AC) / (2.0 * L_AB * L_BC)
    cos_B = max(min(cos_B, 1.0), -1.0)
    ang_B = math.degrees(math.acos(cos_B))

    # 5) Ângulo em C pela soma dos ângulos internos
    ang_C = 180.0 - ang_A - ang_B

    # 6) Área (1/2 * b * c * sen(A))
    area = 0.5 * L_AB * L_AC * math.sin(ang_A_rad)

    coords = {"A": A, "B": B, "C": C}
    angulos = {"A": ang_A, "B": ang_B, "C": ang_C}
    return coords, angulos, area


def figuras_triangulo_por_serie(res: pd.DataFrame, df_par: pd.DataFrame):
    """
    Usa:
      - res: linha a linha, com DH_med_m (por série) e EST, PV
      - df_par: médias por par (para detectar os 3 pontos)
    Se há exatamente 3 pontos, monta triângulo por série apenas com DH + cossenos.
    Retorna:
      pontos: [A,B,C] (nomes reais, ex.: ["P1","P2","P3"])
      figuras: { serie: {"coords":..., "angulos":..., "area":..., "DH":...} }
    """
    pontos = detectar_tres_pontos(df_par)
    if len(pontos) != 3:
        return None, {}

    A, B, C = pontos  # apenas ordenação alfabética

    res_serie = numerar_series_por_estacao(res)
    dh_por_serie = obter_dh_triangulo_por_serie(res_serie, [A, B, C])

    if not dh_por_serie:
        return pontos, {}

    figuras: Dict[int, Dict] = {}
    for s, dhs in dh_por_serie.items():
        DH_AB = dhs["AB"]
        DH_BC = dhs["BC"]
        DH_AC = dhs["AC"]

        coords, angulos, area = montar_triangulo_dh(DH_AB, DH_BC, DH_AC)
        figuras[s] = {
            "coords": coords,
            "angulos": angulos,
            "area": area,
            "DH": dhs,
        }

    return [A, B, C], figuras


# ==================== Desenho do triângulo médio (a partir de coordenadas médias) ====================

def angulo_interno(p_a, p_b, p_c) -> float:
    """Ângulo interno em B, formado pelos segmentos BA e BC."""
    ax, ay = p_a
    bx, by = p_b
    cx, cy = p_c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_ang = max(min(dot / (n1 * n2), 1.0), -1.0)
    return math.degrees(math.acos(cos_ang))


def desenhar_poligono_selecionavel(coords: Dict[str, Tuple[float, float]]):
    """Triângulo escolhido pelo usuário com base nas coordenadas médias (opcional)."""
    if len(coords) < 3:
        st.info("Coordenadas insuficientes para formar um triângulo.")
        return
    pontos = sorted(coords.keys())
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        p_a = st.selectbox("Vértice A (polígono médio)", options=pontos, index=0, key="tri_pt_a")
    with col_sel2:
        opcoes_b = [p for p in pontos if p != p_a]
        p_b = st.selectbox("Vértice B (polígono médio)", options=opcoes_b, index=0, key="tri_pt_b")
    with col_sel3:
        opcoes_c = [p for p in pontos if p not in (p_a, p_b)]
        if not opcoes_c:
            st.info("Selecione A e B diferentes para disponibilizar um C.")
            return
        p_c = st.selectbox("Vértice C (polígono médio)", options=opcoes_c, index=0, key="tri_pt_c")

    A = coords[p_a]
    B = coords[p_b]
    C = coords[p_c]

    dAB = math.hypot(B[0] - A[0], B[1] - A[1])
    dBC = math.hypot(C[0] - B[0], C[1] - B[1])
    dCA = math.hypot(A[0] - C[0], A[1] - C[1])

    ang_A = angulo_interno(B, A, C)
    ang_B = angulo_interno(A, B, C)
    ang_C = angulo_interno(A, C, B)

    xs = [A[0], B[0], C[0], A[0]]
    ys = [A[1], B[1], C[1], A[1]]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, "-o", color="#8B0000", lw=2.3, markersize=8)

    ax.text(A[0], A[1], f" {p_a}", fontsize=10, color="#111827")
    ax.text(B[0], B[1], f" {p_b}", fontsize=10, color="#111827")
    ax.text(C[0], C[1], f" {p_c}", fontsize=10, color="#111827")

    def meio(p, q):
        return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)

    mAB = meio(A, B)
    mBC = meio(B, C)
    mCA = meio(C, A)

    ax.text(mAB[0], mAB[1], f"{dAB:.3f} m", fontsize=9, color="#990000")
    ax.text(mBC[0], mBC[1], f"{dBC:.3f} m", fontsize=9, color="#990000")
    ax.text(mCA[0], mCA[1], f"{dCA:.3f} m", fontsize=9, color="#990000")

    ax.text(A[0], A[1], f"\n∠{p_a} ≈ {ang_A:.2f}°", fontsize=9, color="#1f2937")
    ax.text(B[0], B[1], f"\n∠{p_b} ≈ {ang_B:.2f}°", fontsize=9, color="#1f2937")
    ax.text(C[0], C[1], f"\n∠{p_c} ≈ {ang_C:.2f}°", fontsize=9, color="#1f2937")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.set_title(f"Triângulo {p_a}-{p_b}-{p_c} (coordenadas do polígono médio)")
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)


# ==================== CSS / Cabeçalho / Upload ====================

CUSTOM_CSS = """
<style>
body, .stApp { background: radial-gradient(circle at top left,#fcecea 0%,#f9f1f1 28%,#f4f4f4 55%,#eceff1 100%); color:#111827; font-family:"Trebuchet MS",system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }
.main-card{background:linear-gradient(145deg,rgba(255,255,255,0.98) 0%,#fdf7f7 40%,#ffffff 100%);border-radius:22px;padding:1.8rem 2.1rem 1.4rem 2.1rem;border:1px solid rgba(148,27,37,0.20);box-shadow:0 22px 46px rgba(15,23,42,0.23),0 0 0 1px rgba(15,23,42,0.04);max-width:1280px;margin:1.2rem auto 2.0rem auto;}
.ufpe-top-bar{width:100%;min-height:10px;border-radius:0 0 16px 16px;background:linear-gradient(90deg,#4b0000 0%,#7e0000 30%,#b30000 60%,#4b0000 100%);margin-bottom:1.0rem;}
.ufpe-header-text{font-size:0.8rem;line-height:1.18rem;text-transform:uppercase;color:#111827;}
.ufpe-separator{border:none;border-top:1px solid rgba(148,27,37,0.35);margin:0.8rem 0 1.0rem 0;}
.app-title{font-size:2.0rem;font-weight:800;letter-spacing:0.03em;display:flex;align-items:center;gap:0.65rem;margin-bottom:0.35rem;color:#7f0000;}
.app-title span.icon{font-size:2.4rem;}
.app-subtitle{font-size:0.96rem;color:#374151;margin-bottom:1.0rem;}
.section-title{font-size:1.05rem;font-weight:700;margin-top:1.7rem;margin-bottom:0.6rem;display:flex;align-items:center;gap:0.4rem;color:#8b0000;text-transform:uppercase;letter-spacing:0.05em;}
.section-title span.dot{width:9px;height:9px;border-radius:999px;background:radial-gradient(circle at 30% 30%,#ffffff 0%,#ffbdbd 35%,#7f0000 90%);}
.helper-box{border-radius:14px;padding:0.7rem 0.9rem;background:linear-gradient(135deg,#fff5f5 0%,#ffe7e7 40%,#fffafa 100%);border:1px solid rgba(148,27,37,0.38);font-size:0.85rem;color:#374151;margin-bottom:0.8rem;}
.footer-text{font-size:0.75rem;color:#6b7280;}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{background:linear-gradient(145deg,#ffffff 0%,#f9fafb 50%,#fffdfd 100%) !important;border-radius:14px;border:1px solid rgba(148,27,37,0.22);box-shadow:0 14px 28px rgba(15,23,42,0.10);}
:root{color-scheme:light;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def cabecalho_ufpe():
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown('<div class="ufpe-top-bar"></div>', unsafe_allow_html=True)
        col_logo, col_info = st.columns([1, 5])
        with col_logo:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/8/85/Bras%C3%A3o_da_UFPE.png",
                width=95,
            )
        with col_info:
            st.markdown(
                """
                <div class="ufpe-header-text">
                    <div><strong>UNIVERSIDADE FEDERAL DE PERNAMBUCO</strong></div>
                    <div>DECART — Departamento de Engenharia Cartográfica</div>
                    <div>LATOP — Laboratório de Topografia</div>
                    <div>Curso: <strong>Engenharia Cartográfica e Agrimensura</strong></div>
                    <div>Disciplina: <strong>Equipamentos de Medição</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('<hr class="ufpe-separator">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.text_input("Professor(a)", value="")
            st.text_input("Local", value="")
        with col2:
            st.text_input("Equipamento", value="")
            st.text_input("Patrimônio", value="")
        with col3:
            st.date_input("Data", format="DD/MM/YYYY")
        st.markdown('<hr class="ufpe-separator">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="app-title">
                <span class="icon">📐</span>
                <span>Calculadora de Ângulos e Distâncias</span>
            </div>
            <div class="app-subtitle">
                Médias de Hz e Z, DH, Hz Ré/Vante, polígono médio com azimute de referência
                e triângulo por série construído apenas com DH da série e teorema dos cossenos.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="helper-box">
                <b>Modelo esperado de planilha:</b><br>
                Colunas: <code>EST</code>, <code>PV</code>,
                <code>Hz_PD</code>, <code>Hz_PI</code>,
                <code>Z_PD</code>, <code>Z_PI</code>,
                <code>DI_PD</code>, <code>DI_PI</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def secao_modelo_e_upload():
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>1. Modelo de dados (Hz, Z e DI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    template_df = pd.DataFrame(
        {
            "EST": ["P1", "P1"],
            "PV": ["P2", "P3"],
            "Hz_PD": ["145°47'33\"", "167°29'03\""],
            "Hz_PI": ["325°47'32\"", "347°29'22\""],
            "Z_PD": ["89°48'20\"", "89°36'31\""],
            "Z_PI": ["270°12'00\"", "270°23'32\""],
            "DI_PD": [25.365, 26.285],
            "DI_PI": [25.365, 26.285],
        }
    )
    excel_bytes = io.BytesIO()
    template_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_medicao_direcoes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        "Envie a planilha preenchida (EST, PV, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI)",
        type=["xlsx", "xls", "csv"],
    )
    return uploaded


def processar_upload(uploaded):
    if uploaded is None:
        return None
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

    st.success(f"Arquivo '{uploaded.name}' carregado ({len(raw_df)} linhas).")

    df_valid, erros = validar_dataframe(raw_df)
    st.subheader("Pré-visualização dos dados importados")
    st.dataframe(df_valid[REQUIRED_COLS], use_container_width=True)

    if erros:
        st.error("Não foi possível calcular diretamente devido aos seguintes problemas:")
        for e in erros:
            st.markdown(f"- {e}")
        return None
    else:
        return df_valid[REQUIRED_COLS].copy()


# ==================== Seção de cálculos ====================

def secao_calculos(df_uso: pd.DataFrame):
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>3. Cálculos de Hz, Z e distâncias</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3.1 Linha a linha
    res = calcular_linha_a_linha(df_uso)
    st.markdown("##### Tabela linha a linha (cada série PD/PI)")
    cols_linha = [
        "EST",
        "PV",
        "Hz_PD",
        "Hz_PI",
        "Hz_med_DMS",
        "Z_PD",
        "Z_PI",
        "DH_PD_m",
        "DH_PI_m",
        "DH_med_m",
    ]
    df_linha = res[cols_linha].copy()
    for c in ["DH_PD_m", "DH_PI_m", "DH_med_m"]:
        df_linha[c] = df_linha[c].apply(
            lambda x: f"{x:.3f}".replace(".", ".") if pd.notna(x) else ""
        )
    st.dataframe(df_linha, use_container_width=True)

    # 3.2 Médias por par
    df_par = agregar_por_par(res)

    st.markdown("##### Distâncias médias simétricas entre pontos (diagnóstico)")
    df_dist = tabela_distancias_medias_simetricas(df_par)
    st.dataframe(df_dist, use_container_width=True)

    st.markdown("##### Medição Angular Horizontal")
    tab_hz_par, tab_hz_re_vante = construir_tabela_hz_com_re_vante(df_par)
    st.markdown("**Médias por par (EST–PV):**")
    st.dataframe(tab_hz_par.drop(columns=["Hz_med_deg_par"]), use_container_width=True)
    st.markdown("**Hz Ré/Vante e ângulo reduzido (por estação):**")
    st.dataframe(tab_hz_re_vante, use_container_width=True)

    st.markdown("##### Medição Angular Vertical/Zenital")
    tab_z = tabela_medicao_angular_vertical(df_par)
    st.dataframe(tab_z, use_container_width=True)

    # 4. Polígono médio com azimute de referência (opcional)
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Azimute de referência e polígono médio (opcional)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # escolha padrão da primeira estação encontrada
    if len(df_par["EST"].unique()) > 0:
        est_inicio = str(df_par["EST"].iloc[0])
    else:
        est_inicio = "P1"

    # tenta pegar um PV qualquer associado a essa estação
    sub = df_par[df_par["EST"] == est_inicio]
    if not sub.empty:
        est_segundo = str(sub["PV"].iloc[0])
    else:
        est_segundo = "P2"

    st.markdown(
        f"""
        Informe o <b>azimute conhecido</b> da direção <code>{est_inicio} → {est_segundo}</code>
        (em graus, 0° no Norte, sentido horário). Esse azimute será usado como referência
        para alinhar as direções médias e obter um croqui aproximado (polígono médio).
        """,
        unsafe_allow_html=True,
    )

    az_ref = st.number_input(
        f"Azimute conhecido de {est_inicio} → {est_segundo} (graus, 0 ≤ Az < 360)",
        min_value=0.0,
        max_value=359.9999,
        value=0.0,
        step=0.0001,
    )

    df_par_az = calcular_azimutes_corrigidos(df_par, az_ref, est_inicio, est_segundo)

    st.markdown("**Direções médias com azimute corrigido:**")
    df_show_az = df_par_az[["EST", "PV", "Hz_med_DMS_par", "Az_corrigido_DMS", "DH_med_m_par"]].copy()
    df_show_az.rename(
        columns={
            "Hz_med_DMS_par": "Hz Médio (PD/PI)",
            "Az_corrigido_DMS": "Azimute corrigido",
            "DH_med_m_par": "DH médio (m)",
        },
        inplace=True,
    )
    st.dataframe(df_show_az, use_container_width=True)

    df_coords, coords_dict = calcular_coordenadas(df_par_az, est_inicio)
    st.markdown("**Coordenadas aproximadas (origem na primeira estação detectada):**")
    st.dataframe(df_coords, use_container_width=True)

    st.markdown("**Triângulo selecionável com base no polígono médio:**")
    desenhar_poligono_selecionavel(coords_dict)

    # 5. Triângulo por série (apenas DH da série + cossenos)
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>5. Triângulo por série (DH da série + teorema dos cossenos)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Se os dados contiverem exatamente <b>3 pontos distintos</b> (estações/visadas),
        o app reconhece que se trata de um triângulo e, para cada série,
        monta essa figura apenas com as distâncias horizontais (DH) da própria série,
        usando o <i>teorema dos cossenos</i> para obter os ângulos internos.
        Nenhum azimute real ou coordenada de terreno é utilizado aqui —
        é um croqui puramente geométrico das medições.
        """,
        unsafe_allow_html=True,
    )

    pontos_tri, figuras = figuras_triangulo_por_serie(res, df_par)

    if pontos_tri is None or not figuras:
        st.info(
            "Não foi possível montar o triângulo por série. "
            "Verifique se há exatamente três pontos distintos e DH para todos os lados em cada série."
        )
    else:
        A, B, C = pontos_tri
        st.markdown(f"**Triângulo reconhecido pelos pontos:** `{A}`, `{B}`, `{C}`")

        series_disponiveis = sorted(figuras.keys())
        serie_escolhida = st.selectbox(
            "Escolha a série para visualizar o triângulo correspondente:",
            options=series_disponiveis,
            format_func=lambda s: f"Série {s}",
        )

        dados = figuras[serie_escolhida]
        coords_t = dados["coords"]
        ang_t = dados["angulos"]
        dhs_t = dados["DH"]
        area_t = dados["area"]

        # Desenho do triângulo A–B–C
        P_A = coords_t["A"]
        P_B = coords_t["B"]
        P_C = coords_t["C"]
        xs_t = [P_A[0], P_B[0], P_C[0], P_A[0]]
        ys_t = [P_A[1], P_B[1], P_C[1], P_A[1]]

        fig_t, ax_t = plt.subplots()
        ax_t.plot(xs_t, ys_t, "-o", color="#8B0000", lw=2.3, markersize=8)
        ax_t.text(P_A[0], P_A[1], f" {A}", fontsize=10, color="#111827")
        ax_t.text(P_B[0], P_B[1], f" {B}", fontsize=10, color="#111827")
        ax_t.text(P_C[0], P_C[1], f" {C}", fontsize=10, color="#111827")

        ax_t.set_aspect("equal", "box")
        ax_t.set_xlabel("x (unid. geométrica)")
        ax_t.set_ylabel("y (unid. geométrica)")
        ax_t.set_title(
            f"Triângulo da Série {serie_escolhida} ({A}–{B}–{C}) "
            f"— construído com DH da série + cossenos"
        )
        ax_t.grid(True, linestyle="--", alpha=0.3)

        st.pyplot(fig_t)

        st.markdown("**Lados do triângulo (DH da série):**")
        df_lados_t = pd.DataFrame(
            {
                "Lado": [f"{A}–{B}", f"{B}–{C}", f"{A}–{C}"],
                "DH da série (m)": [
                    round(dhs_t["AB"], 3),
                    round(dhs_t["BC"], 3),
                    round(dhs_t["AC"], 3),
                ],
            }
        )
        st.dataframe(df_lados_t, use_container_width=True)

        st.markdown("**Ângulos internos do triângulo (°):**")
        df_ang_t = pd.DataFrame(
            {
                "Vértice": [A, B, C],
                "Ângulo interno (°)": [
                    round(ang_t["A"], 4),
                    round(ang_t["B"], 4),
                    round(ang_t["C"], 4),
                ],
            }
        )
        st.dataframe(df_ang_t, use_container_width=True)

        st.markdown(f"**Área do triângulo (série {serie_escolhida}):** `{area_t:.4f} m²`")


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>UFPE_v4.0 — triângulo por série usando apenas DH da série e teorema dos cossenos;
            Hz/Z médios, Ré/Vante, polígono médio com azimute de referência.</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==================== Fluxo principal ====================

cabecalho_ufpe()
uploaded = secao_modelo_e_upload()
df_uso = processar_upload(uploaded)

if df_uso is not None:
    secao_calculos(df_uso)

rodape()
