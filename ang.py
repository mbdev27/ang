# app.py
# Calculadora de Ângulos e Distâncias — UFPE
# Hz/Z/DH + Ré/Vante + Polígono com azimute de referência + Figuras por série (ordem dinâmica)

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

# Este mapa (Ré/Vante) continua fixo nos rótulos P1,P2,P3.
# Se você quiser que ele também fique 100% dinâmico, podemos adaptar depois.
RE_VANTE_MAP: Dict[str, Tuple[str, str]] = {
    "P1": ("P2", "P3"),  # (Ré, Vante)
    "P2": ("P1", "P3"),
    "P3": ("P1", "P2"),
}

# ==================== Funções auxiliares de ângulo ====================

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


# ==================== Pré-processamento do DataFrame ====================

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

    invalid_rows_hz: List[int] = []
    invalid_rows_z: List[int] = []
    invalid_rows_di: List[int] = []

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
        return pd.Series(out)

    df_par = res.groupby(["EST", "PV"], as_index=False).apply(agg_par)

    df_par["Hz_med_deg_par"] = df_par.apply(
        lambda r: mean_direction_two(r["Hz_PD_med_deg"], r["Hz_PI_med_deg"]), axis=1
    )
    df_par["Hz_med_DMS_par"] = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    zpd_par_rad = df_par["Z_PD_med_deg"] * np.pi / 180.0
    zpi_par_rad = df_par["Z_PI_med_deg"] * np.pi / 180.0

    df_par["DH_PD_m_par"] = np.abs(
        df_par["DI_PD_med_m"] * np.sin(zpd_par_rad)
    ).round(3)
    df_par["DN_PD_m_par"] = np.abs(
        df_par["DI_PD_med_m"] * np.cos(zpd_par_rad)
    ).round(3)
    df_par["DH_PI_m_par"] = np.abs(
        df_par["DI_PI_med_m"] * np.sin(zpi_par_rad)
    ).round(3)
    df_par["DN_PI_m_par"] = np.abs(
        df_par["DI_PI_med_m"] * np.cos(zpi_par_rad)
    ).round(3)

    df_par["DH_med_m_par"] = np.abs(
        (df_par["DH_PD_m_par"] + df_par["DH_PI_m_par"]) / 2.0
    ).round(3)
    df_par["DN_med_m_par"] = np.abs(
        (df_par["DN_PD_m_par"] + df_par["DN_PI_m_par"]) / 2.0
    ).round(3)

    return df_par


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


# ==================== Cálculo de coordenadas com azimute de referência ====================

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
        dh = r["DH_med_m_par"]
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
    ang = math.degrees(math.acos(cos_ang))
    return ang


def desenhar_poligono_selecionavel(coords: Dict[str, Tuple[float, float]]):
    if len(coords) < 3:
        st.info("Coordenadas insuficientes para formar um triângulo.")
        return

    pontos_disponiveis = sorted(coords.keys())

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        p_a = st.selectbox(
            "Vértice A do triângulo (coordenadas do polígono médio)",
            options=pontos_disponiveis,
            index=0,
            key="tri_pt_a",
        )
    with col_sel2:
        opcoes_b = [p for p in pontos_disponiveis if p != p_a]
        p_b = st.selectbox(
            "Vértice B do triângulo (coordenadas do polígono médio)",
            options=opcoes_b,
            index=0,
            key="tri_pt_b",
        )
    with col_sel3:
        opcoes_c = [p for p in pontos_disponiveis if p not in (p_a, p_b)]
        if len(opcoes_c) == 0:
            st.info("Selecione A e B diferentes para disponibilizar um C.")
            return
        p_c = st.selectbox(
            "Vértice C do triângulo (coordenadas do polígono médio)",
            options=opcoes_c,
            index=0,
            key="tri_pt_c",
        )

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

    st.markdown("**Resumo dos lados (geométricos) do triângulo selecionado:**")
    df_lados = pd.DataFrame(
        {
            "Lado": [f"{p_a}–{p_b}", f"{p_b}–{p_c}", f"{p_c}–{p_a}"],
            "Distância geométrica (m)": [round(dAB, 3), round(dBC, 3), round(dCA, 3)],
        }
    )
    st.dataframe(df_lados, use_container_width=True)

    st.markdown("**Ângulos internos no triângulo selecionado:**")
    df_ang = pd.DataFrame(
        {
            "Vértice": [p_a, p_b, p_c],
            "Ângulo interno (°)": [round(ang_A, 2), round(ang_B, 2), round(ang_C, 2)],
        }
    )
    st.dataframe(df_ang, use_container_width=True)


# ========= NOVO: regra das séries para montar figuras (sem médias) =========

def numerar_series_por_estacao(res_linha: pd.DataFrame) -> pd.DataFrame:
    df = res_linha.copy()
    df["SERIE"] = df.groupby("EST").cumcount().astype(int) + 1
    return df


def detectar_ordem_estacoes(df_uso: pd.DataFrame, minimo: int = 3) -> List[str]:
    """
    Detecta a ordem das estações pela primeira aparição em EST (sem repetir).
    Ex.: se EST aparece como P1, P3, P2, P1, P3..., ordem = [P1, P3, P2].
    """
    ordem = []
    for est in df_uso["EST"]:
        if est not in ordem:
            ordem.append(est)
        if len(ordem) >= minimo:
            break
    return ordem


def figuras_por_serie_triangulo_generico(
    res_linha_serie: pd.DataFrame,
    az_ref: float,
    ordem_estacoes: List[str],
):
    """
    Gera figuras por série para um triângulo definido pela ordem_estacoes[0:3].
    - Vértices: E1, E2, E3 nessa ordem.
    - Para cada série s:
        E1 -> E2
        E2 -> E3
        E3 -> E1
    - Usa Hz_med_deg + az_ref (referência: E1->E2 primeira ocorrência) e DH_med_m.
    Retorna: dict serie -> {coords, df_lados, df_ang, area}
    """
    if len(ordem_estacoes) < 3:
        return {}

    E1, E2, E3 = ordem_estacoes[:3]
    resultados = {}

    estacoes_necessarias = {E1, E2, E3}
    if not estacoes_necessarias.issubset(set(res_linha_serie["EST"].unique())):
        return resultados

    # Direção de referência para alinhar azimute: E1 -> E2 (primeira ocorrência)
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

        # convenção: E1->E2, E2->E3, E3->E1
        l_E1_E2 = lE1.iloc[[0]]
        l_E2_E3 = lE2.iloc[[0]]
        l_E3_E1 = lE3.iloc[[0]]

        az_E1_E2, dh_E1_E2 = linha_para_az_e_dh(l_E1_E2)
        az_E2_E3, dh_E2_E3 = linha_para_az_e_dh(l_E2_E3)
        az_E3_E1, dh_E3_E1 = linha_para_az_e_dh(l_E3_E1)

        P1 = (0.0, 0.0)  # vértice E1 na origem

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


# ==================== CSS e identidade visual UFPE ====================

CUSTOM_CSS = """
<style>
body, .stApp {
    background:
        radial-gradient(circle at top left, #fcecea 0%, #f9f1f1 28%, #f4f4f4 55%, #eceff1 100%);
    color: #111827;
    font-family: "Trebuchet MS", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
/* (CSS idêntico ao anterior, omitido por brevidade na explicação) */
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== Cabeçalho UFPE ====================

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
                Cálculo da média das direções Hz, ângulo vertical (Z), distâncias horizontais,
                Hz reduzido (Ré/Vante) e coordenadas aproximadas do polígono.
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


# ==================== Seção modelo e upload ====================

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


# ==================== Seção de cálculos principais ====================

def secao_calculos(df_uso: pd.DataFrame):
    # Detecta ordem das estações automaticamente
    ordem_estacoes = detectar_ordem_estacoes(df_uso, minimo=3)

    st.markdown(
        f"**Ordem detectada das estações (para construção de figuras):** {', '.join(ordem_estacoes)}",
    )

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>3. Cálculos de Hz, Z e distâncias (linha a linha e por par)</span>
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

    st.markdown("##### Medição Angular Horizontal")
    tab_hz_par, tab_hz_re_vante = construir_tabela_hz_com_re_vante(df_par)

    st.markdown("**Médias por par (EST–PV):**")
    st.dataframe(tab_hz_par.drop(columns=["Hz_med_deg_par"]), use_container_width=True)

    st.markdown("**Hz Ré/Vante e ângulo reduzido (por estação):**")
    st.dataframe(tab_hz_re_vante, use_container_width=True)

    st.markdown("##### Medição Angular Vertical/Zenital")
    tab_z = tabela_medicao_angular_vertical(df_par)
    st.dataframe(tab_z, use_container_width=True)

    # ==================== Azimute de referência e polígono médio ====================
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
        para alinhar todas as direções medidas.
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

    # ==================== Figuras por série (regra das leituras) ====================
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>5. Figuras por série (sem médias, regra das leituras)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Para cada série <code>s</code> (1ª, 2ª, 3ª, ...), monta-se uma figura usando
        <b>apenas</b> as leituras de índice <code>s</code> de cada estação.
        A ordem dos vértices é dada pela primeira aparição das estações na tabela:
        1ª estação, depois 2ª, depois 3ª (para o triângulo).
        """,
        unsafe_allow_html=True,
    )

    res_serie = numerar_series_por_estacao(res)
    figuras = figuras_por_serie_triangulo_generico(res_serie, az_ref, ordem_estacoes)

    if not figuras:
        st.info(
            "Não foi possível montar figuras por série. "
            "É necessário ter pelo menos três estações distintas com séries compatíveis."
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

        ests = list(coords_t.keys())
        A_id, B_id, C_id = ests[0], ests[1], ests[2]
        A = coords_t[A_id]
        B = coords_t[B_id]
        C = coords_t[C_id]

        xs_t = [A[0], B[0], C[0], A[0]]
        ys_t = [A[1], B[1], C[1], A[1]]

        fig_t, ax_t = plt.subplots()
        ax_t.plot(xs_t, ys_t, "-o", color="#8B0000", lw=2.3, markersize=8)
        ax_t.text(A[0], A[1], f" {A_id}", fontsize=10, color="#111827")
        ax_t.text(B[0], B[1], f" {B_id}", fontsize=10, color="#111827")
        ax_t.text(C[0], C[1], f" {C_id}", fontsize=10, color="#111827")

        ax_t.set_aspect("equal", "box")
        ax_t.set_xlabel("E (m)")
        ax_t.set_ylabel("N (m)")
        ax_t.set_title(f"Triângulo da Série {serie_escolhida} ({A_id}–{B_id}–{C_id}, sem médias)")
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
            Versão do app: <code>UFPE_v2.4 — Hz/Z, Ré/Vante, azimute de referência, polígono médio e figuras por série com ordem automática das estações.</code>.
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
