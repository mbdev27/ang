# app.py
# Calculadora de Ângulos e Distâncias — UFPE
# Hz/Z/DH + Ré/Vante + Polígono com azimute de referência e identidade visual UFPE

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

# Convenção implícita do seu calcula_poligono.py (Ré e Vante por estação)
RE_VANTE_MAP: Dict[str, Tuple[str, str]] = {
    "P1": ("P2", "P3"),  # (Ré, Vante)
    "P2": ("P1", "P3"),
    "P3": ("P1", "P2"),
}

# ==================== Funções auxiliares de ângulo ====================

def parse_angle_to_decimal(value: str) -> float:
    """
    Converte string de ângulo em DMS (145°47′33″, 145°47'33", 145 47 33)
    ou decimal ("145.7925") para graus decimais.
    Retorna NaN se não conseguir converter.
    """
    if value is None:
        return float("nan")

    s = str(value).strip()
    if s == "":
        return float("nan")

    # 1) tenta decimal simples
    try:
        if all(ch.isdigit() or ch in ".,-+" for ch in s):
            return float(s.replace(",", "."))
    except Exception:
        pass

    # 2) normaliza símbolos DMS para espaços
    for ch in ["°", "º", "'", "’", "´", "′", '"', "″"]:
        s = s.replace(ch, " ")

    # vírgula como ponto
    s = s.replace(",", ".")

    parts = s.split()
    parts = [p for p in parts if p != ""]
    if len(parts) == 0:
        return float("nan")

    # 3) interpreta como D, M, S
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
    """
    Converte graus decimais para string DMS com segundos inteiros: 145°47'34"
    """
    if angle_deg is None or math.isnan(angle_deg):
        return ""
    sign = "-" if angle_deg < 0 else ""
    a = abs(angle_deg)

    d = int(a)
    m_f = (a - d) * 60
    m = int(m_f)
    s_f = (m_f - m) * 60

    # arredonda segundos para inteiro
    s = int(round(s_f))

    # ajusta “estouro” de 60"
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d += 1

    return f"{sign}{d:02d}°{m:02d}'{s:02d}\""


def mean_direction_two(a_deg: float, b_deg: float) -> float:
    """
    Média vetorial de duas direções em graus.
    """
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
    """
    Média vetorial de uma lista (Series) de ângulos em graus.
    """
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
    """
    Harmoniza nomes de colunas vindos de planilhas diversas para:
    EST, PV, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI.
    """
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
    """
    Normaliza colunas e verifica colunas obrigatórias + campos válidos.
    """
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
    """
    Converte ângulos, distâncias, e calcula Hz_médio, DH/DN linha a linha.
    """
    res = df_uso.copy()

    # Ângulos em decimal
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    # Distâncias inclinadas
    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    z_pd_rad = res["Z_PD_deg"] * np.pi / 180.0
    z_pi_rad = res["Z_PI_deg"] * np.pi / 180.0

    # DH / DN (3 casas decimais)
    res["DH_PD_m"] = np.abs(res["DI_PD_m"] * np.sin(z_pd_rad)).round(3)
    res["DN_PD_m"] = np.abs(res["DI_PD_m"] * np.cos(z_pd_rad)).round(3)
    res["DH_PI_m"] = np.abs(res["DI_PI_m"] * np.sin(z_pi_rad)).round(3)
    res["DN_PI_m"] = np.abs(res["DI_PI_m"] * np.cos(z_pi_rad)).round(3)

    # Hz médio linha a linha
    res["Hz_med_deg"] = res.apply(
        lambda r: mean_direction_two(r["Hz_PD_deg"], r["Hz_PI_deg"]), axis=1
    )
    res["Hz_med_DMS"] = res["Hz_med_deg"].apply(decimal_to_dms)

    # DH/DN médios linha a linha (3 casas)
    res["DH_med_m"] = np.abs((res["DH_PD_m"] + res["DH_PI_m"]) / 2.0).round(3)
    res["DN_med_m"] = np.abs((res["DN_PD_m"] + res["DN_PI_m"]) / 2.0).round(3)

    return res


def agregar_por_par(res: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega em um DataFrame por par EST–PV.
    """

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

    # Hz médio por par
    df_par["Hz_med_deg_par"] = df_par.apply(
        lambda r: mean_direction_two(r["Hz_PD_med_deg"], r["Hz_PI_med_deg"]), axis=1
    )
    df_par["Hz_med_DMS_par"] = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    # DH/DN médios por par (3 casas)
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
    """
    Constrói tabela Horizontal com:
      EST, PV, Hz_PD, Hz_PI, Hz_Médio,
      e tabela de Hz_Ré, Hz_Vante, α (Ré → Vante) por estação.
    """
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

    # Tabela de Ré/Vante por estação
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
        alpha = hz_va - hz_re
        alpha = (alpha + 360.0) % 360.0

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
    """
    Tabela Vertical:
    EST, PV, Z_PD, Z_PI, Z Corrigido, Média das Séries.
    Usa: Z = (Z_PD_med - Z_PI_med) / 2 + 180°
    """
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
    """
    ΔE = Dh * sin(az), ΔN = Dh * cos(az)
    az em graus a partir do Norte (0°), sentido horário.
    """
    az_rad = math.radians(az_deg)
    de = dh * math.sin(az_rad)
    dn = dh * math.cos(az_rad)
    return de, dn


def calcular_azimutes_corrigidos(df_par: pd.DataFrame, az_ref_p1p2: float) -> pd.DataFrame:
    """
    Ajusta Hz_med_deg_par para virar azimute, usando az_ref_p1p2 (P1→P2) como referência:
      offset = az_ref_p1p2 - Hz_med(P1→P2)
      Az_corrigido = (Hz_med + offset) mod 360
    """
    df_par = df_par.copy()

    # encontra Hz médio para P1→P2
    mask_p1p2 = (df_par["EST"] == "P1") & (df_par["PV"] == "P2")
    if not mask_p1p2.any():
        # se não tiver P1→P2, apenas trata Hz como se já fosse azimute
        df_par["Az_corrigido_deg"] = df_par["Hz_med_deg_par"] % 360.0
        df_par["Az_corrigido_DMS"] = df_par["Az_corrigido_deg"].apply(decimal_to_dms)
        return df_par

    hz_p1p2 = df_par.loc[mask_p1p2, "Hz_med_deg_par"].iloc[0]
    offset = az_ref_p1p2 - hz_p1p2

    df_par["Az_corrigido_deg"] = (df_par["Hz_med_deg_par"] + offset) % 360.0
    df_par["Az_corrigido_DMS"] = df_par["Az_corrigido_deg"].apply(decimal_to_dms)

    return df_par


def calcular_coordenadas(df_par_az: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Usa Az_corrigido_deg como azimute (graus 0..360) e DH_med_m_par
    para calcular as coordenadas aproximadas dos pontos (P1, P2, P3...).
    Assume P1 = (0,0) e propaga pelas observações.
    """
    coords: Dict[str, Tuple[float, float]] = {}
    coords["P1"] = (0.0, 0.0)  # origem

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

    # Propaga iterativamente
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
    """
    Calcula o ângulo interno no vértice B (A-B-C) em graus.
    """
    ax, ay = p_a
    bx, by = p_b
    cx, cy = p_c

    # vetores BA e BC
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])

    if n1 == 0 or n2 == 0:
        return float("nan")

    cos_ang = dot / (n1 * n2)
    cos_ang = max(min(cos_ang, 1.0), -1.0)
    ang = math.degrees(math.acos(cos_ang))
    return ang


def desenhar_poligono(coords: Dict[str, Tuple[float, float]]):
    """
    Desenha o triângulo P1–P2–P3 (se existirem) com rótulos de lados e ângulos internos.
    """
    must_pts = ["P1", "P2", "P3"]
    if not all(p in coords for p in must_pts):
        st.info("Coordenadas insuficientes para desenhar o triângulo P1–P2–P3.")
        return

    p1 = coords["P1"]
    p2 = coords["P2"]
    p3 = coords["P3"]

    # Distâncias
    d12 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    d23 = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
    d31 = math.hypot(p1[0] - p3[0], p1[1] - p3[1])

    # Ângulos internos
    ang_p1 = angulo_interno(p2, p1, p3)
    ang_p2 = angulo_interno(p1, p2, p3)
    ang_p3 = angulo_interno(p1, p3, p2)

    xs = [p1[0], p2[0], p3[0], p1[0]]
    ys = [p1[1], p2[1], p3[1], p1[1]]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, "-o", color="#8B0000", lw=2.3, markersize=8)

    # rótulos dos pontos
    ax.text(p1[0], p1[1], " P1", fontsize=10, color="#111827")
    ax.text(p2[0], p2[1], " P2", fontsize=10, color="#111827")
    ax.text(p3[0], p3[1], " P3", fontsize=10, color="#111827")

    # rótulos dos lados (meio de cada segmento)
    def meio(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    m12 = meio(p1, p2)
    m23 = meio(p2, p3)
    m31 = meio(p3, p1)

    ax.text(m12[0], m12[1], f"{d12:.3f} m", fontsize=9, color="#990000")
    ax.text(m23[0], m23[1], f"{d23:.3f} m", fontsize=9, color="#990000")
    ax.text(m31[0], m31[1], f"{d31:.3f} m", fontsize=9, color="#990000")

    # rótulos dos ângulos próximos aos vértices
    ax.text(p1[0], p1[1], f"\n∠P1 ≈ {ang_p1:.2f}°", fontsize=9, color="#1f2937")
    ax.text(p2[0], p2[1], f"\n∠P2 ≈ {ang_p2:.2f}°", fontsize=9, color="#1f2937")
    ax.text(p3[0], p3[1], f"\n∠P3 ≈ {ang_p3:.2f}°", fontsize=9, color="#1f2937")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.set_title("Polígono aproximado P1–P2–P3 (distâncias e ângulos internos)")
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)

    # tabela resumo de distâncias e ângulos
    st.markdown("**Resumo dos lados e ângulos internos:**")
    df_ang = pd.DataFrame(
        {
            "Lado": ["P1–P2", "P2–P3", "P3–P1"],
            "Distância (m)": [round(d12, 3), round(d23, 3), round(d31, 3)],
        }
    )
    df_int = pd.DataFrame(
        {
            "Vértice": ["P1", "P2", "P3"],
            "Ângulo interno (°)": [round(ang_p1, 2), round(ang_p2, 2), round(ang_p3, 2)],
        }
    )
    col_lados, col_angs = st.columns(2)
    with col_lados:
        st.dataframe(df_ang, use_container_width=True)
    with col_angs:
        st.dataframe(df_int, use_container_width=True)


# ==================== CSS e identidade visual UFPE ====================

CUSTOM_CSS = """
<style>
body, .stApp {
    background:
        radial-gradient(circle at top left, #fcecea 0%, #f9f1f1 28%, #f4f4f4 55%, #eceff1 100%);
    color: #111827;
    font-family: "Trebuchet MS", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Cartão principal */
.main-card {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.98) 0%, #fdf7f7 40%, #ffffff 100%);
    border-radius: 22px;
    padding: 1.8rem 2.1rem 1.4rem 2.1rem;
    border: 1px solid rgba(148,27,37,0.20);
    box-shadow:
        0 22px 46px rgba(15, 23, 42, 0.23),
        0 0 0 1px rgba(15, 23, 42, 0.04);
    max-width: 1280px;
    margin: 1.2rem auto 2.0rem auto;
}

/* Faixa superior em degradê vermelho */
.ufpe-top-bar {
    width: 100%;
    min-height: 10px;
    border-radius: 0 0 16px 16px;
    background:
        linear-gradient(90deg, #4b0000 0%, #7e0000 30%, #b30000 60%, #4b0000 100%);
    margin-bottom: 1.0rem;
}

/* Texto do cabeçalho institucional */
.ufpe-header-text {
    font-size: 0.8rem;
    line-height: 1.18rem;
    text-transform: uppercase;
    color: #111827;
}
.ufpe-header-text strong {
    letter-spacing: 0.06em;
}

/* Linha separadora */
.ufpe-separator {
    border: none;
    border-top: 1px solid rgba(148,27,37,0.35);
    margin: 0.8rem 0 1.0rem 0;
}

/* Título principal */
.app-title {
    font-size: 2.0rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.35rem;
    color: #7f0000;
}
.app-title span.icon {
    font-size: 2.4rem;
}

/* Subtítulo */
.app-subtitle {
    font-size: 0.96rem;
    color: #374151;
    margin-bottom: 1.0rem;
}

/* Títulos de seção */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 1.7rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #8b0000;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.section-title span.dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background:
        radial-gradient(circle at 30% 30%, #ffffff 0%, #ffbdbd 35%, #7f0000 90%);
}

/* Caixinha de ajuda */
.helper-box {
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
    background:
        linear-gradient(135deg, #fff5f5 0%, #ffe7e7 40%, #fffafa 100%);
    border: 1px solid rgba(148,27,37,0.38);
    font-size: 0.85rem;
    color: #374151;
    margin-bottom: 0.8rem;
}

/* Rodapé */
.footer-text {
    font-size: 0.75rem;
    color: #6b7280;
}

/* Tabelas e dataframes */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background:
        linear-gradient(145deg, #ffffff 0%, #f9fafb 50%, #fffdfd 100%) !important;
    border-radius: 14px;
    border: 1px solid rgba(148,27,37,0.22);
    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.10);
}

[data-testid="stDataFrame"] thead tr {
    background:
        linear-gradient(90deg, #fbe5e7 0%, #fcd7dd 50%, #fbe5e7 100%) !important;
    color: #4b0000 !important;
    font-weight: 700;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: #fdfbfb !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #f3eff0 !important;
}

/* Campos de entrada flutuando sobre fundo */
.stTextInput, .stNumberInput, .stDateInput, .stFileUploader {
    background:
        linear-gradient(135deg, #ffffff 0%, #f9f7f7 40%, #ffffff 100%) !important;
}

/* Forçar identidade independente do tema do navegador */
:root {
    color-scheme: light;
}
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

        # Linha com campos: Professor, Local, Equipamento, Data, Patrimônio
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

        # Título do app
        st.markdown(
            """
            <div class="app-title">
                <span class="icon">📐</span>
                <span>Calculadora de Ângulos e Distâncias</span>
            </div>
            <div class="app-subtitle">
                Cálculo da média das direções Hz, ângulo vertical (Z), distâncias horizontais,
                Hz reduzido (Ré/Vante) e coordenadas aproximadas do polígono P1–P2–P3.
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
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>3. Cálculos de Hz, Z e distâncias (linha a linha e por par)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Linha a linha
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

    # Formata DH com vírgula e 3 casas decimais
    for c in ["DH_PD_m", "DH_PI_m", "DH_med_m"]:
        df_linha[c] = df_linha[c].apply(
            lambda x: f"{x:.3f}".replace(".", ",") if pd.notna(x) else ""
        )

    st.dataframe(df_linha, use_container_width=True)

    # Agregado por par EST–PV
    df_par = agregar_por_par(res)

    # Tabela Horizontal com Ré/Vante
    st.markdown("##### Medição Angular Horizontal")
    st.markdown(
        """
        <b>Fórmulas utilizadas (Hz médio e Hz reduzido)</b><br><br>
        Média das direções (por série PD/PI):<br>
        <span style="font-family: 'DejaVu Sans Mono', monospace;">
        Hz = ( Hz<sub>PD</sub> + Hz<sub>PI</sub> ) / 2 &plusmn; 90&deg;
        </span>
        <br><br>
        com:<br>
        &nbsp;&nbsp;&bull; + se Hz<sub>PD</sub> &gt; Hz<sub>PI</sub><br>
        &nbsp;&nbsp;&bull; &minus; se Hz<sub>PD</sub> &lt; Hz<sub>PI</sub><br><br>
        Cálculo do ângulo entre duas direções (redução entre Ré e Vante):<br>
        <span style="font-family: 'DejaVu Sans Mono', monospace;">
        &alpha; = Hz<sub>Vante</sub> &minus; Hz<sub>R&eacute;</sub>
        </span>
        """,
        unsafe_allow_html=True,
    )

    tab_hz_par, tab_hz_re_vante = construir_tabela_hz_com_re_vante(df_par)

    st.markdown("**Médias por par (EST–PV):**")
    st.dataframe(
        tab_hz_par.drop(columns=["Hz_med_deg_par"]), use_container_width=True
    )

    st.markdown("**Hz Ré/Vante e ângulo reduzido (por estação):**")
    st.dataframe(tab_hz_re_vante, use_container_width=True)

    # Tabela Vertical
    st.markdown("##### Medição Angular Vertical/Zenital")
    st.markdown(
        """
        <b>Fórmula utilizada (Z corrigido)</b><br><br>
        <span style="font-family: 'DejaVu Sans Mono', monospace;">
        Z = ( Z'<sub>PD</sub> &minus; Z'<sub>PI</sub> ) / 2 + 180&deg;
        </span>
        """,
        unsafe_allow_html=True,
    )
    tab_z = tabela_medicao_angular_vertical(df_par)
    st.dataframe(tab_z, use_container_width=True)

    # ==================== Azimute de referência e polígono ====================
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Azimute de referência e polígono P1–P2–P3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Informe o <b>azimute conhecido</b> da direção <code>P1 → P2</code> (em graus, 0° no Norte, sentido horário).
        O programa alinhará o Hz médio dessa direção a esse azimute e aplicará o mesmo ajuste
        às demais direções, gerando coordenadas coerentes com o seu levantamento.
        """,
        unsafe_allow_html=True,
    )

    az_ref_p1p2 = st.number_input(
        "Azimute conhecido de P1 → P2 (graus, 0 ≤ Az < 360)",
        min_value=0.0,
        max_value=359.9999,
        value=0.0,
        step=0.0001,
    )

    df_par_az = calcular_azimutes_corrigidos(df_par, az_ref_p1p2)

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

    df_coords, coords_dict = calcular_coordenadas(df_par_az)

    st.markdown("**Coordenadas aproximadas (origem em P1 = 0,0):**")
    st.dataframe(df_coords, use_container_width=True)

    st.markdown("**Polígono aproximado P1–P2–P3 com lados e ângulos internos:**")
    desenhar_poligono(coords_dict)


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>UFPE_v2.1 — Hz/Z, Ré/Vante, azimute de referência, coordenadas e polígono com identidade visual UFPE.</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # fecha main-card


# ==================== Fluxo principal ====================

cabecalho_ufpe()
uploaded = secao_modelo_e_upload()
df_uso = processar_upload(uploaded)

if df_uso is not None:
    secao_calculos(df_uso)

rodape()
