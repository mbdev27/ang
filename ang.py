# app.py
# UFPE - Calculadora de Ângulos e Distâncias
# Versão sem qualquer referência a Ré/Vante
# - Z é zenital/vertical, com DH = DI × seno(Z)
# - SEQ indica a ordem da medição na série

import io
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ==================== Config página ====================
st.set_page_config(
    page_title="Calculadora de Ângulos e Distâncias | UFPE",
    layout="wide",
    page_icon="📐",
)

# ==================== Parâmetros globais ====================

REQUIRED_COLS_BASE = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]
OPTIONAL_COLS = ["SEQ"]  # SEQ = número da medição na série em cada estação
REQUIRED_COLS_ALL = REQUIRED_COLS_BASE + OPTIONAL_COLS

# ==================== Funções auxiliares de ângulo ====================

def parse_angle_to_decimal(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "":
        return float("nan")
    # decimal direto?
    try:
        if all(ch.isdigit() or ch in ".,-+" for ch in s):
            return float(s.replace(",", "."))
    except Exception:
        pass
    # formato DMS
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


def validar_dataframe(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    erros: List[str] = []
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
    if invalid_rows_seq:
        erros.append(
            "Valores inválidos em SEQ (devem ser inteiros) nas linhas: "
            + ", ".join(map(str, invalid_rows_seq))
            + "."
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


# ==================== Cálculos linha a linha / por par ====================

def calcular_linha_a_linha(df_uso: pd.DataFrame) -> pd.DataFrame:
    res = df_uso.copy()
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    z_pd_rad = res["Z_PD_deg"] * np.pi / 180.0
    z_pi_rad = res["Z_PI_deg"] * np.pi / 180.0

    # Z zenital/vertical: DH = DI × sen(Z)
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


# ==================== Tabelas Hz e Z ====================

def tabela_medicao_angular_horizontal(df_par: pd.DataFrame) -> pd.DataFrame:
    hz_pd_med_dms = df_par["Hz_PD_med_deg"].apply(decimal_to_dms)
    hz_pi_med_dms = df_par["Hz_PI_med_deg"].apply(decimal_to_dms)
    hz_med_dms = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    tab = pd.DataFrame(
        {
            "EST": df_par["EST"],
            "PV": df_par["PV"],
            "Hz PD (médio)": hz_pd_med_dms,
            "Hz PI (médio)": hz_pi_med_dms,
            "Hz Médio (PD/PI)": hz_med_dms,
        }
    )
    return tab


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


# ==================== Distâncias médias simétricas ====================

def tabela_distancias_medias_simetricas(df_par: pd.DataFrame) -> pd.DataFrame:
    aux = df_par[["EST", "PV", "DH_med_m_par"]].copy()
    registros = {}

    for _, row in aux.iterrows():
        a = str(row["EST"])
        b = str(row["PV"])
        if a == b:
            continue
        par = tuple(sorted([a, b]))
        dh = float(row["DH_med_m_par"])
        registros.setdefault(par, []).append(dh)

    linhas = []
    for (a, b), valores in registros.items():
        dh_med = float(np.mean(valores))
        linhas.append({"PontoA": a, "PontoB": b, "DH_media": dh_med})

    df_dist = pd.DataFrame(linhas)
    if not df_dist.empty:
        df_dist.sort_values("DH_media", ascending=False, inplace=True)
    return df_dist


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
                Médias de Hz e Z, DH = DI·sen(Z) (Z zenital/vertical) e diagnósticos de distâncias.
                Versão sem identificação automática de ré/vante, preparada para novas regras geométricas.
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
                <code>DI_PD</code>, <code>DI_PI</code>.<br>
                <b>SEQ</b>: número da sequência (1 = 1ª medição, 2 = 2ª, etc.) em cada estação.
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
            "EST": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "PV":  ["P2", "P3", "P2", "P3", "P2", "P3"],
            "SEQ": [1,    1,    2,    2,    3,    3],
            "Hz_PD": ["145°47'33\"", "167°29'03\"", "145°47'48\"", "167°29'36\"", "145°47'45\"", "165°29'37\""],
            "Hz_PI": ["325°47'32\"", "347°29'22\"", "325°47'28\"", "347°29'18\"", "325°47'17\"", "347°29'09\""],
            "Z_PD":  ["89°48'20\"", "89°36'31\"", "89°48'21\"", "89°36'29\"", "89°48'05\"", "89°36'11\""],
            "Z_PI":  ["270°12'00\"", "270°23'32\"", "270°11'53\"", "270°23'36\"", "270°12'07\"", "270°23'43\""],
            "DI_PD": [25.365, 26.285, 25.365, 26.284, 25.365, 26.284],
            "DI_PI": [25.365, 26.285, 25.365, 26.285, 25.365, 26.285],
        }
    )
    excel_bytes = io.BytesIO()
    template_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_medicao_direcoes_com_SEQ.xlsx",
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
        st.error("Não foi possível calcular diretamente devido aos seguintes problemas:")
        for e in erros:
            st.markdown(f"- {e}")
        return None
    else:
        cols_use = [c for c in REQUIRED_COLS_ALL if c in df_valid.columns]
        return df_valid[cols_use].copy()


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
    cols_linha = ["EST", "PV"]
    if "SEQ" in res.columns:
        cols_linha.append("SEQ")
    cols_linha += [
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
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )
    st.dataframe(df_linha, use_container_width=True)

    # 3.2 Médias por par
    df_par = agregar_por_par(res)

    st.markdown("##### Distâncias médias simétricas entre pontos (diagnóstico)")
    df_dist = tabela_distancias_medias_simetricas(df_par)
    st.dataframe(df_dist, use_container_width=True)

    st.markdown("##### Medição Angular Horizontal (médias por par)")
    tab_hz = tabela_medicao_angular_horizontal(df_par)
    st.dataframe(tab_hz, use_container_width=True)

    st.markdown("##### Medição Angular Vertical/Zenital")
    tab_z = tabela_medicao_angular_vertical(df_par)
    st.dataframe(tab_z, use_container_width=True)

    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Próximo passo: regras de formação de figuras</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        A partir daqui, vamos definir como o app deve usar os <b>Hz médios</b> e as <b>distâncias DH</b>
        (por série, por estação, etc.) para montar triângulos, quadriláteros ou outros polígonos,
        segundo as regras que você definir (estação base, ângulos entre visadas, etc.).
        """,
        unsafe_allow_html=True,
    )


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>UFPE_v5.1 — Hz/Z/DH com SEQ, sem identificação de ré/vante; regras geométricas a definir.</code>.
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
