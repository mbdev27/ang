# app.py
# UFPE - Calculadora de Ângulos e Distâncias (Método das Direções)

import io
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Calculadora de Ângulos e Distâncias | UFPE",
    layout="wide",
    page_icon="📐",
)

REQUIRED_COLS_BASE = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]
OPTIONAL_COLS = ["SEQ"]
REQUIRED_COLS_ALL = REQUIRED_COLS_BASE + OPTIONAL_COLS

# =====================================================================
#  Funções de ângulo
# =====================================================================

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
    if not parts:
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

# =====================================================================
#  Normalização / Validação
# =====================================================================

def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    colmap = {}
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

# =====================================================================
#  Cálculos linha a linha
# =====================================================================

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

# =====================================================================
#  Tabelas por série (Hz / Z)
# =====================================================================

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
        df.loc[mask, "Hz_reduzido_deg"] = (
            (df.loc[mask, "Hz_med_deg"] - ref) % 360.0
        )

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

# =====================================================================
#  Distâncias simétricas e 7ª tabela resumo
# =====================================================================

def tabela_distancias_medias_simetricas(res: pd.DataFrame) -> pd.DataFrame:
    aux = res[["EST", "PV", "DH_med_m"]].copy()
    registros = {}

    for _, row in aux.iterrows():
        a = str(row["EST"])
        b = str(row["PV"])
        if a == b:
            continue
        par = tuple(sorted([a, b]))
        dh = float(row["DH_med_m"])
        registros.setdefault(par, []).append(dh)

    linhas = []
    for (a, b), valores in registros.items():
        dh_med = float(np.mean(valores))
        linhas.append({"PontoA": a, "PontoB": b, "DH_media": dh_med})

    df_dist = pd.DataFrame(linhas)
    if not df_dist.empty:
        df_dist.sort_values("DH_media", ascending=False, inplace=True)
    return df_dist


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
            "Média das séries": "Média das Séries (Hz)",
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
                "Média das Séries (Hz)",
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
                "Média das Séries (Hz)",
                "Z Corrigido",
                "Média Z das séries",
                "DH Médio (m)",
            ]
        ]

    return resumo

# =====================================================================
#  Triângulo – cálculos e seleção automática
# =====================================================================

def _angulo_interno(a, b, c):
    try:
        if a <= 0 or b <= 0 or c <= 0:
            return float("nan")
        cosA = (b**2 + c**2 - a**2) / (2 * b * c)
        cosA = max(-1.0, min(1.0, cosA))
        return math.degrees(math.acos(cosA))
    except Exception:
        return float("nan")


def calcular_triangulo_duas_linhas(res: pd.DataFrame, idx1: int, idx2: int):
    if idx1 == idx2:
        return None
    if idx1 < 0 or idx1 >= len(res) or idx2 < 0 or idx2 >= len(res):
        return None

    r1 = res.iloc[idx1]
    r2 = res.iloc[idx2]

    est1, est2 = str(r1["EST"]), str(r2["EST"])
    pv1, pv2 = str(r1["PV"]), str(r2["PV"])

    if est1 != est2:
        return None
    if pv1 == pv2:
        return None

    est = est1
    b = float(r1["DH_med_m"])
    c = float(r2["DH_med_m"])
    hz1 = float(r1["Hz_med_deg"])
    hz2 = float(r2["Hz_med_deg"])

    alpha_deg = (hz2 - hz1) % 360.0
    if alpha_deg > 180.0:
        alpha_deg = 360.0 - alpha_deg

    a = math.sqrt(
        b**2 + c**2 - 2 * b * c * math.cos(math.radians(alpha_deg))
    )

    A_int = _angulo_interno(a, b, c)
    B_int = _angulo_interno(b, a, c)
    C_int = _angulo_interno(c, a, b)

    s = (a + b + c) / 2.0
    area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))

    return {
        "EST": est,
        "PV1": pv1,
        "PV2": pv2,
        "b_EST_PV1": b,
        "c_EST_PV2": c,
        "a_PV1_PV2": a,
        "alpha_EST_deg": alpha_deg,
        "ang_EST_deg": A_int,
        "ang_PV1_deg": B_int,
        "ang_PV2_deg": C_int,
        "area_m2": area,
    }


def selecionar_linhas_por_estacao_e_conjunto(
    res: pd.DataFrame, estacao_letra: str, conjunto: str
) -> Optional[Tuple[int, int]]:
    """
    Usa apenas a ORDEM das linhas e as combinações EST/PV para definir
    1ª, 2ª e 3ª leituras para cada estação (A,B,C).

    - Estação A (P1):
        1ª leitura: EST=P2 e PV in {P3, P1}  (B>C + B>A)
        2ª leitura: EST=P1 e PV in {P2, P3}  (A>B + A>C)  [2º par]
        3ª leitura: mesmo padrão, 3º par
    - Estação B (P2):
        1ª,2ª,3ª: EST=P2 e PV in {P3, P1} (B>C + B>A), 1º,2º,3º pares
    - Estação C (P3):
        1ª,2ª,3ª: EST=P3 e PV in {P1, P2} (C>A + C>B), 1º,2º,3º pares
    """
    letra_to_p = {"A": "P1", "B": "P2", "C": "P3"}
    est_ref = letra_to_p.get(estacao_letra)
    if est_ref is None:
        return None

    ordem = {"1ª leitura": 1, "2ª leitura": 2, "3ª leitura": 3}[conjunto]

    df = res.reset_index(drop=False).rename(columns={"index": "_idx_orig"})

    # Define filtro principal conforme estação e regra
    if est_ref == "P1":  # Estação A
        if ordem == 1:
            # 1ª leitura: B>C + B>A  => EST=P2, PV in {P3, P1}
            mask = (df["EST"] == "P2") & (df["PV"].isin(["P3", "P1"]))
        else:
            # 2ª e 3ª: A>B + A>C => EST=P1, PV in {P2, P3}
            mask = (df["EST"] == "P1") & (df["PV"].isin(["P2", "P3"]))
    elif est_ref == "P2":  # Estação B
        # B>C + B>A
        mask = (df["EST"] == "P2") & (df["PV"].isin(["P3", "P1"]))
    else:  # est_ref == "P3", Estação C
        # C>A + C>B
        mask = (df["EST"] == "P3") & (df["PV"].isin(["P1", "P2"]))

    cand = df[mask].sort_values(by="_idx_orig")
    if len(cand) < 2:
        return None

    # agrupar em pares na ordem em que aparecem
    cand = cand.reset_index(drop=True)
    cand["par_id"] = cand.index // 2  # 0,0,1,1,2,2...

    # qual par queremos? 1ª->0, 2ª->1, 3ª->2
    par_desejado = ordem - 1
    par = cand[cand["par_id"] == par_desejado]
    if len(par) < 2:
        return None

    idxs = par["_idx_orig"].tolist()[:2]
    return int(idxs[0]), int(idxs[1])

# =====================================================================
#  Plotagem do triângulo (formato croqui)
# =====================================================================

def plotar_triangulo_info(info):
    est = info["EST"]
    pv1 = info["PV1"]
    pv2 = info["PV2"]

    b = info["b_EST_PV1"]
    c = info["c_EST_PV2"]
    a = info["a_PV1_PV2"]

    x_est, y_est = 0.0, 0.0
    x_pv2, y_pv2 = c, 0.0

    if c == 0:
        x_pv1, y_pv1 = b, 0.0
    else:
        x_pv1 = (b**2 - a**2 + c**2) / (2 * c)
        arg = max(b**2 - x_pv1**2, 0.0)
        y_pv1 = math.sqrt(arg)

    xs = [x_est, x_pv1, x_pv2, x_est]
    ys = [y_est, y_pv1, y_pv2, y_est]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, "-o", color="#7f0000")
    ax.set_aspect("equal", "box")

    ax.text(x_est, y_est, f" {est}", fontsize=10, color="#111827")
    ax.text(x_pv1, y_pv1, f" {pv1}", fontsize=10, color="#111827")
    ax.text(x_pv2, y_pv2, f" {pv2}", fontsize=10, color="#111827")

    ax.text((x_est + x_pv1) / 2, (y_est + y_pv1) / 2,
            f"{b:.3f} m", color="#374151", fontsize=9)
    ax.text((x_est + x_pv2) / 2, (y_est + y_pv2) / 2,
            f"{c:.3f} m", color="#374151", fontsize=9)
    ax.text((x_pv1 + x_pv2) / 2, (y_pv1 + y_pv2) / 2,
            f"{a:.3f} m", color="#374151", fontsize=9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Representação do triângulo em planta")

    st.pyplot(fig)

# =====================================================================
#  CSS, cabeçalho, upload, seções de cálculo
# =====================================================================

# (toda a parte de CSS, cabecalho_ufpe, secao_modelo_e_upload,
#  processar_upload, secao_calculos e rodape fica igual à da
#  última versão que você já testou – para economizar espaço,
#  não repito aqui, mas é só colar essas funções acima, trocando
#  apenas a parte da seção 8 para usar a nova seleção automática.)

    # ---------- 8. Triângulo com seleção automática ----------
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>8. Triângulo selecionado (conjunto automático de medições)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        estacao_op = st.selectbox("Estação (A, B, C)", ["A", "B", "C"])
    with col_b:
        conjunto_op = st.selectbox(
            "Conjunto de leituras",
            ["1ª leitura", "2ª leitura", "3ª leitura"],
        )

    st.markdown(
        "O programa seleciona automaticamente o par de leituras adequado "
        "para formar o triângulo, conforme as regras definidas para cada estação."
    )

    if st.button("Gerar triângulo"):
        pares = selecionar_linhas_por_estacao_e_conjunto(res, estacao_op, conjunto_op)
        if pares is None:
            st.error(
                "Não foi possível encontrar duas leituras compatíveis para "
                f"Estação {estacao_op} e {conjunto_op}. "
                "Verifique se a coluna SEQ e os PVs estão preenchidos como no modelo."
            )
        else:
            idx1, idx2 = pares
            info = calcular_triangulo_duas_linhas(res, idx1, idx2)
            if info is None:
                st.error("Falha ao calcular o triângulo a partir das leituras selecionadas.")
            else:
                est = info["EST"]
                pv1 = info["PV1"]
                pv2 = info["PV2"]

                st.markdown(
                    f"**Triângulo formado automaticamente por {est}, {pv1} e {pv2} "
                    f"({conjunto_op} na Estação {estacao_op}).**"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Lados (m):**")
                    st.markdown(
                        f"- {est}–{pv1}: `{info['b_EST_PV1']:.3f}` m\n"
                        f"- {est}–{pv2}: `{info['c_EST_PV2']:.3f}` m\n"
                        f"- {pv1}–{pv2}: `{info['a_PV1_PV2']:.3f}` m"
                    )
                    st.markdown("**Ângulos internos:**")
                    st.markdown(
                        f"- Em {est}: `{decimal_to_dms(info['ang_EST_deg'])}`\n"
                        f"- Em {pv1}: `{decimal_to_dms(info['ang_PV1_deg'])}`\n"
                        f"- Em {pv2}: `{decimal_to_dms(info['ang_PV2_deg'])}`"
                    )
                    st.markdown(
                        f"**Área do triângulo:** `{info['area_m2']:.3f}` m²"
                    )
                with col2:
                    plotar_triangulo_info(info)


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>UFPE_v11.0 — seleção automática de conjuntos de leituras por estação (A,B,C) e triângulo em planta.</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


cabecalho_ufpe()
uploaded = secao_modelo_e_upload()
df_uso = processar_upload(uploaded)

if df_uso is not None:
    secao_calculos(df_uso)

rodape()
