# app.py
# Versão: UFPE_v3.2 — ordem dos vértices baseada nas distâncias médias (maior, intermediária, menor)

import io
import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Calculadora de Ângulos e Distâncias | UFPE",
    layout="wide",
    page_icon="📐",
)

REQUIRED_COLS = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]

RE_VANTE_MAP: Dict[str, Tuple[str, str]] = {
    "P1": ("P2", "P3"),
    "P2": ("P1", "P3"),
    "P3": ("P1", "P2"),
}

# ========= Ângulos / helpers =========

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


# ========= Normalização / validação =========

def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
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


# ========= Cálculos linha a linha / por par =========

def calcular_linha_a_linha(df_uso: pd.DataFrame) -> pd.DataFrame:
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


# ========= Tabela Hz / Z =========

def construir_tabela_hz_com_re_vante(df_par: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


# ========= Distâncias médias simétricas e ordem por distância =========

def tabela_distancias_medias_simetricas(df_par: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de df_par (com DH_med_m_par por EST,PV),
    monta uma tabela de distâncias médias simétricas por par de pontos:
    chave = {A,B} (independe da ordem), valor = média das DH de A->B e B->A (se existirem).
    """
    # tabela auxiliar com DH
    aux = df_par[["EST", "PV", "DH_med_m_par"]].copy()
    registros = {}

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
        df_dist.sort_values("DH_media", ascending=False, inplace=True)  # maior → menor
    return df_dist


def detectar_ordem_estacoes_por_distancia(df_par: pd.DataFrame) -> List[str]:
    """
    Para o caso de 3 pontos:
    - calcula as distâncias DH médias simétricas
    - ordena da maior para a menor
    - retorna uma ordem coerente de vértices [E1,E2,E3],
      de forma que:
      - lado maior seja E1–E3 (caso tipo P1–P3),
      - intermediário envolva E1 (tipo P1–P2),
      - menor feche triângulo (tipo P2–P3).
    Em geral, encontra uma sequência dos 3 pontos que usa todos os lados.
    """
    df_dist = tabela_distancias_medias_simetricas(df_par)
    if df_dist.empty or len(df_dist) < 3:
        # fallback: ordem lexicográfica pelos nomes encontrados
        pontos = sorted(set(df_par["EST"]).union(set(df_par["PV"])))
        return pontos[:3]

    # pontos distintos
    pontos = set(df_dist["PontoA"]).union(set(df_dist["PontoB"]))
    if len(pontos) != 3:
        # se tiver mais ou menos de 3, mantém fallback simples
        return sorted(pontos)[:3]

    # pegar os 3 pares (maior, intermediário, menor)
    pares = list(df_dist[["PontoA", "PontoB"]].itertuples(index=False, name=None))
    # maior distância
    pA_maior, pB_maior = pares[0]
    # o terceiro vértice é o ponto que não está no par de maior DH
    terceiro = list(pontos - {pA_maior, pB_maior})[0]

    # agora temos 3 pontos: pA_maior, pB_maior, terceiro
    # vamos definir ordem: [pA_maior, terceiro, pB_maior]
    # isso é análogo a: P1, P3, P2 (maior lado = P1–P3)
    ordem = [pA_maior, terceiro, pB_maior]
    return ordem


# ========= Coordenadas com azimute de referência =========

def delta_from_azimuth(az_deg: float, dh: float) -> Tuple[float, float]:
    az_rad = math.radians(az_deg)
    de = dh * math.sin(az_rad)
    dn = dh * math.cos(az_rad)
    return de, dn


def calcular_azimutes_corrigidos(df_par: pd.DataFrame, az_ref: float, est_inicio: str, est_segundo: str) -> pd.DataFrame:
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


def angulo_interno(p_a, p_b, p_c) -> float:
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


# ========= Séries / figuras por série (usando DH da série) =========

def numerar_series_por_estacao(res_linha: pd.DataFrame) -> pd.DataFrame:
    df = res_linha.copy()
    df["SERIE"] = df.groupby("EST").cumcount().astype(int) + 1
    return df


def figuras_por_serie_triangulo_por_distancia(
    res_linha_serie: pd.DataFrame,
    az_ref: float,
    ordem_estacoes: List[str],
):
    """
    Triângulo por série, ordem dos vértices veio de detectar_ordem_estacoes_por_distancia().
    ordem_estacoes = [E1, E2, E3], onde
        - maior DH é E1–E3
        - intermediária E1–E2
        - menor E2–E3 (na analogia P1–P3, P1–P2, P2–P3).
    Para cada série s:
        E1->E2, E2->E3, E3->E1
    Usa Hz_med_deg da série + offset (az_ref - Hz(E1->E2 série 1)) e DH_med_m da série.
    """
    if len(ordem_estacoes) != 3:
        return {}

    E1, E2, E3 = ordem_estacoes
    resultados = {}

    # direção de referência E1->E2 (primeira ocorrência de qualquer série)
    linha_ref = res_linha_serie[(res_linha_serie["EST"] == E1) & (res_linha_serie["PV"] == E2)].head(1)
    if linha_ref.empty:
        return resultados
    hz_ref = linha_ref["Hz_med_deg"].iloc[0]
    offset = az_ref - hz_ref

    def linha_para_az_e_dh(linha):
        hz = linha["Hz_med_deg"].iloc[0]
        dh = linha["DH_med_m"].iloc[0]
        az = (hz + offset) % 360.0
        return az, dh

    n_series = int(res_linha_serie.groupby("EST")["SERIE"].max().min())

    for s in range(1, n_series + 1):
        lE1 = res_linha_serie[(res_linha_serie["EST"] == E1) & (res_linha_serie["SERIE"] == s)]
        lE2 = res_linha_serie[(res_linha_serie["EST"] == E2) & (res_linha_serie["SERIE"] == s)]
        lE3 = res_linha_serie[(res_linha_serie["EST"] == E3) & (res_linha_serie["SERIE"] == s)]
        if lE1.empty or lE2.empty or lE3.empty:
            continue

        l_E1 = lE1.iloc[[0]]
        l_E2 = lE2.iloc[[0]]
        l_E3 = lE3.iloc[[0]]

        az_E1_E2, dh_E1_E2 = linha_para_az_e_dh(l_E1)
        az_E2_E3, dh_E2_E3 = linha_para_az_e_dh(l_E2)
        az_E3_E1, dh_E3_E1 = linha_para_az_e_dh(l_E3)

        P1 = (0.0, 0.0)
        de12, dn12 = delta_from_azimuth(az_E1_E2, dh_E1_E2)
        P2 = (P1[0] + de12, P1[1] + dn12)
        de23, dn23 = delta_from_azimuth(az_E2_E3, dh_E2_E3)
        P3 = (P2[0] + de23, P2[1] + dn23)

        coords = {E1: P1, E2: P2, E3: P3}

        d12 = math.hypot(P2[0] - P1[0], P2[1] - P1[1])
        d23 = math.hypot(P3[0] - P2[0], P3[1] - P2[1])
        d31 = math.hypot(P1[0] - P3[0], P1[1] - P3[1])

        df_lados = pd.DataFrame(
            {
                "Lado": [f"{E1}–{E2}", f"{E2}–{E3}", f"{E3}–{E1}"],
                "Distância geométrica (m)": [round(d12, 3), round(d23, 3), round(d31, 3)],
                "DH da série (m)": [
                    round(dh_E1_E2, 3),
                    round(dh_E2_E3, 3),
                    round(dh_E3_E1, 3),
                ],
            }
        )

        ang_1 = angulo_interno(P3, P1, P2)
        ang_2 = angulo_interno(P1, P2, P3)
        ang_3 = angulo_interno(P2, P3, P1)

        df_ang = pd.DataFrame(
            {
                "Vértice": [E1, E2, E3],
                "Ângulo interno (°)": [round(ang_1, 4), round(ang_2, 4), round(ang_3, 4)],
            }
        )

        x1, y1 = P1
        x2, y2 = P2
        x3, y3 = P3
        area = abs(
            x1 * (y2 - y3)
            + x2 * (y3 - y1)
            + x3 * (y1 - y2)
        ) / 2.0

        resultados[s] = {
            "coords": coords,
            "df_lados": df_lados,
            "df_ang": df_ang,
            "area": area,
        }

    return resultados


# ========= Desenho polígono médio (triângulo livre) =========

def desenhar_poligono_selecionavel(coords: Dict[str, Tuple[float, float]]):
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


# ========= CSS / cabeçalho / upload (idêntico aos anteriores, apenas resumido) =========

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
                Médias de Hz e Z, DH, Hz Ré/Vante, azimute de referência, polígono médio
                e triângulos por série com ordem automática baseada nas distâncias.
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


# ========= Seção de cálculos =========

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

    df_par = agregar_por_par(res)

    # tabela de distâncias médias simétricas e ordem dos vértices
    df_dist = tabela_distancias_medias_simetricas(df_par)
    st.markdown("##### Distâncias médias simétricas entre pontos (maior → menor)")
    st.dataframe(df_dist, use_container_width=True)

    ordem_estacoes = detectar_ordem_estacoes_por_distancia(df_par)
    st.markdown(
        f"**Ordem dos vértices detectada pela distância (triângulo):** {', '.join(ordem_estacoes)}"
    )

    st.markdown("##### Medição Angular Horizontal")
    tab_hz_par, tab_hz_re_vante = construir_tabela_hz_com_re_vante(df_par)
    st.markdown("**Médias por par (EST–PV):**")
    st.dataframe(tab_hz_par.drop(columns=["Hz_med_deg_par"]), use_container_width=True)
    st.markdown("**Hz Ré/Vante e ângulo reduzido (por estação):**")
    st.dataframe(tab_hz_re_vante, use_container_width=True)

    st.markdown("##### Medição Angular Vertical/Zenital")
    tab_z = tabela_medicao_angular_vertical(df_par)
    st.dataframe(tab_z, use_container_width=True)

    # ===== Azimute referência / polígono médio =====
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Azimute de referência e polígono médio</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(ordem_estacoes) >= 2:
        est_inicio, est_segundo = ordem_estacoes[0], ordem_estacoes[1]
    else:
        est_inicio, est_segundo = "P1", "P2"

    st.markdown(
        f"""
        Informe o <b>azimute conhecido</b> da direção <code>{est_inicio} → {est_segundo}</code>
        (em graus, 0° no Norte, sentido horário). Esse azimute será usado como referência
        para alinhar todas as direções médias.
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

    # ===== Figuras por série (triângulo) =====
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>5. Triângulos por série (usando DH da série)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        O triângulo é definido pelos pontos: <code>{' → '.join(ordem_estacoes)}</code>.<br>
        A ordem foi inferida a partir das distâncias médias: lado maior, intermediário, menor.
        """,
        unsafe_allow_html=True,
    )

    res_serie = numerar_series_por_estacao(res)
    figuras = figuras_por_serie_triangulo_por_distancia(res_serie, az_ref, ordem_estacoes)

    if not figuras:
        st.info(
            "Não foi possível montar triângulos por série. "
            "É necessário ter pelo menos três estações com séries compatíveis."
        )
    else:
        series_disponiveis = sorted(figuras.keys())
        serie_escolhida = st.selectbox(
            "Escolha a série para visualizar o triângulo correspondente:",
            options=series_disponiveis,
            format_func=lambda s: f"Série {s}",
        )

        dados = figuras[serie_escolhida]
        coords_t = dados["coords"]
        df_lados_t = dados["df_lados"]
        df_ang_t = dados["df_ang"]
        area_t = dados["area"]

        E1, E2, E3 = ordem_estacoes
        P1, P2, P3 = coords_t[E1], coords_t[E2], coords_t[E3]

        xs_t = [P1[0], P2[0], P3[0], P1[0]]
        ys_t = [P1[1], P2[1], P3[1], P1[1]]

        fig_t, ax_t = plt.subplots()
        ax_t.plot(xs_t, ys_t, "-o", color="#8B0000", lw=2.3, markersize=8)

        ax_t.text(P1[0], P1[1], f" {E1}", fontsize=10, color="#111827")
        ax_t.text(P2[0], P2[1], f" {E2}", fontsize=10, color="#111827")
        ax_t.text(P3[0], P3[1], f" {E3}", fontsize=10, color="#111827")

        ax_t.set_aspect("equal", "box")
        ax_t.set_xlabel("E (m)")
        ax_t.set_ylabel("N (m)")
        ax_t.set_title(f"Triângulo da Série {serie_escolhida} ({E1}–{E2}–{E3}) — DH da série")
        ax_t.grid(True, linestyle="--", alpha=0.3)

        st.pyplot(fig_t)

        st.markdown("**Lados do triângulo (geométricos vs. DH da série):**")
        st.dataframe(df_lados_t, use_container_width=True)

        st.markdown("**Ângulos internos do triângulo da série:**")
        st.dataframe(df_ang_t, use_container_width=True)

        st.markdown(f"**Área da figura da série {serie_escolhida}:** `{area_t:.4f} m²`")


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>UFPE_v3.2 — ordem dos vértices baseada nas distâncias médias (maior, intermediária, menor),
            triângulos por série com DH da própria série.</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ========= Fluxo principal =========

cabecalho_ufpe()
uploaded = secao_modelo_e_upload()
df_uso = processar_upload(uploaded)

if df_uso is not None:
    secao_calculos(df_uso)

rodape()
