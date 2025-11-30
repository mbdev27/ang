# app.py
# UFPE - Calculadora de Ângulos e Distâncias (Método das Direções)

import io
import math
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Calculadora de Ângulos e Distâncias | UFPE",
    layout="wide",
    page_icon="📐",
)

REQUIRED_COLS_BASE = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]
OPTIONAL_COLS = ["SEQ"]  # Número da série (usado só internamente)
REQUIRED_COLS_ALL = REQUIRED_COLS_BASE + OPTIONAL_COLS

# ==================== Funções de ângulo ====================


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


# ==================== Normalização / Validação ====================


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


# ==================== Cálculos linha a linha ====================


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


# ==================== Tabelas por série (modelo dos slides) ====================


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


# ==================== 7ª Tabela – Resumo final ====================


def tabela_resumo_final(res: pd.DataFrame, renomear_para_letras: bool = True) -> pd.DataFrame:
    """
    Tabela-resumo:

    EST | PV | Hz Médio | Hz Reduzido | Média das Séries (Hz) | Z Corrigido | Média das Séries (Z) | DH Médio

    Se renomear_para_letras=True, apenas substitui:
      P1 -> A, P2 -> B, P3 -> C
    (sem criar novas linhas ou novo mapeamento).
    """
    # Reaproveita resultados das tabelas de Hz e Z
    tab_hz = tabela_hz_por_serie(res)  # Hz Médio, Hz Reduzido, Média das séries (Hz)
    tab_z = tabela_z_por_serie(res)    # Z Corrigido, Média das séries (Z)

    hz_cols = ["Estação", "Ponto Visado", "Hz Médio", "Hz Reduzido", "Média das séries"]
    z_cols = ["Estação", "Ponto Visado", "Z Corrigido", "Média das séries"]

    df_hz = tab_hz[hz_cols].copy()
    df_z = tab_z[z_cols].copy()
    df_z = df_z.rename(columns={"Média das séries": "Média Z das séries"})

    resumo = pd.merge(
        df_hz,
        df_z,
        on=["Estação", "Ponto Visado"],
        how="outer",
    )

    # DH médio por (EST, PV)
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

    # Seleciona e renomeia colunas de saída
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

    # ===== Renomear apenas P1, P2, P3 -> A, B, C =====
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
        # Mantém os nomes originais
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


# ==================== CSS / Layout ====================

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
                Médias das direções horizontais (Hz) e medição angular vertical/zenital
                seguindo os modelos dos exemplos de sala (Método das Direções).
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="helper-box">
                <b>Modelo esperado de planilha:</b><br>
                Colunas: <code>EST</code>, <code>PV</code>, <code>SEQ</code>,
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
            <span>1. Modelo de dados (Hz, Z, DI e SEQ)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    template_df = pd.DataFrame(
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
    excel_bytes = io.BytesIO()
    template_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_medicao_direcoes_exemplos.xlsx",
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
        "Envie a planilha preenchida (EST, PV, SEQ, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI)",
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
    cols_to_show = [c for c in REQUIRED_COLS_ALL if c in df_valid.columns]
    st.dataframe(df_valid[cols_to_show], use_container_width=True)

    if erros:
        st.error("Não foi possível calcular devido aos seguintes problemas:")
        for e in erros:
            st.markdown(f"- {e}")
        return None
    else:
        cols_use = [c for c in REQUIRED_COLS_ALL if c in df_valid.columns]
        return df_valid[cols_use].copy()


def secao_calculos(df_uso: pd.DataFrame):
    st.markdown(
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
    st.dataframe(df_linha, use_container_width=True)

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Medição Angular Horizontal (modelo do Exemplo 1)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_hz = tabela_hz_por_serie(res)
    st.dataframe(tab_hz, use_container_width=True)

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>5. Medição Angular Vertical / Zenital (modelo do Exemplo 2)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_z = tabela_z_por_serie(res)
    st.dataframe(tab_z, use_container_width=True)

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>6. Distâncias médias horizontais simétricas (diagnóstico)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    df_dist = tabela_distancias_medias_simetricas(res)
    st.dataframe(df_dist, use_container_width=True)

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>7. Tabela resumo (Hz, Z e DH)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sempre aplicar mapeamento P1->A, P2->B, P3->C
    resumo = tabela_resumo_final(res, renomear_para_letras=True)
    st.dataframe(resumo, use_container_width=True)


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>UFPE_v9.0 — Inclui tabela resumo com Hz, Z e DH, mapeando P1→A, P2→B, P3→C.</code>.
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
