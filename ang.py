# app.py
# Média das Direções (Hz) + DH/DN - UFPE + Croqui P1-P2-P3
# Rode com: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math
import matplotlib.pyplot as plt

# -------------------- Configuração da página --------------------
st.set_page_config(
    page_title="Média das Direções (Hz) — Estação Total | UFPE",
    layout="wide",
    page_icon="📐"
)

# -------------------- Estilos customizados --------------------
CUSTOM_CSS = """
<style>
body, .stApp {
    background: radial-gradient(circle at top left, #faf5f5 0%, #f7f5f5 45%, #f4f4f4 100%);
    color: #111827;
    font-family: "Trebuchet MS", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
.main-card {
    background: linear-gradient(145deg, #ffffff 0%, #fdfbfb 40%, #ffffff 100%);
    border-radius: 18px;
    padding: 1.8rem 2.1rem;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 18px 40px rgba(15, 23, 42, 0.22),
        0 0 0 1px rgba(15, 23, 42, 0.03);
}
.ufpe-top-bar {
    width: 100%;
    height: 8px;
    border-radius: 0 0 14px 14px;
    background: linear-gradient(90deg, #5b0000 0%, #990000 52%, #5b0000 100%);
    margin-bottom: 0.9rem;
}
.ufpe-header-text {
    font-size: 0.78rem;
    line-height: 1.15rem;
    text-transform: uppercase;
    color: #111827;
}
.ufpe-header-text strong {
    letter-spacing: 0.04em;
}
.ufpe-separator {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 0.8rem 0 1.0rem 0;
}
.app-title {
    font-size: 2.0rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.35rem;
    color: #111827;
}
.app-title span.icon {
    font-size: 2.4rem;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #4b5563;
    margin-bottom: 0.9rem;
}
.section-title {
    font-size: 1.02rem;
    font-weight: 600;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #111827;
}
.section-title span.dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 30%, #ffffff 0%, #990000 40%, #111827 100%);
}
.helper-box {
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    background: linear-gradient(135deg, #fdf2f2 0%, #fff5f5 40%, #fdf2f2 100%);
    border: 1px solid rgba(153, 0, 0, 0.25);
    font-size: 0.83rem;
    color: #4b5563;
    margin-bottom: 0.7rem;
}
.footer-text {
    font-size: 0.75rem;
    color: #6b7280;
}
.stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid #990000;
    background: linear-gradient(135deg, #b00000, #730000);
    color: white;
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.45rem 0.95rem;
    box-shadow: 0 8px 18px rgba(128,0,0,0.35);
}
.stDownloadButton > button:hover {
    border-color: #111827;
    background: linear-gradient(135deg, #111827, #4b0000);
}

/* Tabelas */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background: linear-gradient(145deg, #ffffff 0%, #f9fafb 60%, #ffffff 100%) !important;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
}
[data-testid="stDataFrame"] thead tr {
    background: linear-gradient(90deg, #f5e6e8 0%, #fdf2f2 100%) !important;
    color: #5b101d !important;
    font-weight: 600;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: #fafafa !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #f3f0f0 !important;
}

/* Inputs */
.stTextInput label, .stFileUploader label {
    font-size: 0.86rem;
    font-weight: 600;
    color: #374151;
}
.stTextInput input {
    background: linear-gradient(145deg, #ffffff, #f9fafb) !important;
    color: #111827 !important;
    border-radius: 999px;
    border: 1px solid #d4d4d4;
    padding: 0.35rem 0.8rem;
    font-size: 0.9rem;
}
.stTextInput input:focus {
    border-color: #a32a36 !important;
    box-shadow: 0 0 0 1px rgba(163,42,54,0.25);
}

/* Botões */
.stButton button {
    background: linear-gradient(135deg, #a32a36, #7d1220) !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 0.4rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    box-shadow: 0 6px 16px rgba(125,18,32,0.25);
}
.stButton button:hover {
    background: linear-gradient(135deg, #7d1220, #5a0d18) !important;
    box-shadow: 0 4px 12px rgba(90,13,24,0.35);
}

/* Uploader PT-BR */
[data-testid="stFileUploaderDropzone"] > div > div {
    font-size: 0.9rem;
}
[data-testid="stFileUploaderDropzone"]::before {
    content: "Arraste e solte o arquivo aqui ou";
    display: block;
    text-align: center;
    margin-bottom: 0.25rem;
    color: #374151;
    font-size: 0.88rem;
}
[data-testid="stFileUploaderDropzone"] button {
    color: #ffffff !important;
    background: linear-gradient(135deg, #a32a36, #7d1220) !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 0.2rem 0.9rem;
    font-size: 0.85rem;
}
[data-testid="stFileUploaderDropzone"] button::before {
    content: "Escolher arquivo";
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Cabeçalho ----------------------
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

        col_prof, col_local, col_equip, col_data, col_patr = st.columns(
            [1.6, 1.4, 1.6, 1.1, 1.2]
        )
        professor = col_prof.text_input("Professor", value="")
        local = col_local.text_input("Local", value="")
        equipamento = col_equip.text_input("Equipamento", value="")
        data_trabalho = col_data.text_input("Data", value="")
        patrimonio = col_patr.text_input("Patrimônio", value="")

    st.markdown('<hr class="ufpe-separator">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-title">
            <span class="icon">📐</span>
            <span>Média das Direções (Hz) — Estação Total</span>
        </div>
        <div class="app-subtitle">
            Cálculo da média das direções Hz, distâncias horizontais / diferenças de nível
            e croqui plano do triângulo P1–P2–P3.
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
            <code>DI_PD</code>, <code>DI_PI</code>.<br>
            Ângulos em <b>DMS</b> (ex.: 145°47'33") ou <b>decimal</b> (ex.: 145.7925).<br>
            Distâncias inclinadas em <b>metros</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- 1. Modelo Excel ----------
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>1. Modelo de dados (Hz, Z e DI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_df = pd.DataFrame({
        "EST": ["P1", "P1"],
        "PV": ["P2", "P3"],
        "Hz_PD": ["145°47'33\"", "167°29'03\""],
        "Hz_PI": ["325°47'32\"", "347°29'22\""],
        "Z_PD":  ["89°48'20\"", "89°36'31\""],
        "Z_PI":  ["270°12'00\"", "270°23'32\""],
        "DI_PD": [25.365, 26.285],
        "DI_PI": [25.365, 26.285],
    })

    excel_bytes = io.BytesIO()
    template_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_estacao_total_ufpe.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_model"
    )

    # -------- 2. Upload ----------
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
        help="Use o modelo disponibilizado acima para evitar problemas de formatação."
    )

# -------------------- Funções auxiliares --------------------------
angle_re = re.compile(r"(-?\d+)[^\d\-]+(\d+)[^\d\-]+(\d+(?:[.,]\d+)?)")
num_re = re.compile(r"^-?\d+(?:[.,]\d+)?$")

def parse_angle_to_decimal(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = (
        s.replace("º", "°")
         .replace("\u2019", "'")
         .replace("\u201d", '"')
         .replace(":", " ")
         .replace("\t", " ")
    )
    s = re.sub(r"\s+", " ", s)
    m = angle_re.search(s)
    if m:
        deg = float(m.group(1))
        minu = float(m.group(2))
        sec = float(m.group(3).replace(",", "."))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu/60.0 + sec/3600.0)
    if num_re.match(s.replace(" ", "")):
        return float(s.replace(",", "."))
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s)
    if len(nums) == 3:
        deg, minu, sec = nums
        deg = float(deg)
        minu = float(minu)
        sec = float(str(sec).replace(",", "."))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu/60.0 + sec/3600.0)
    return np.nan

def decimal_to_dms(angle):
    if pd.isna(angle):
        return ""
    a = float(angle) % 360.0
    g = int(math.floor(a))
    m_f = (a - g) * 60.0
    m = int(math.floor(m_f))
    s_f = (m_f - m) * 60.0
    s = int(round(s_f))
    if s >= 60:
        s = 0
        m += 1
    if m >= 60:
        m = 0
        g += 1
    return f"{g:02d}°{m:02d}'{s:02d}\""

def mean_direction_two(a_deg, b_deg):
    if pd.isna(a_deg) or pd.isna(b_deg):
        return np.nan
    a_rad = math.radians(a_deg)
    b_rad = math.radians(b_deg)
    x = math.cos(a_rad) + math.cos(b_rad)
    y = math.sin(a_rad) + math.sin(b_rad)
    if x == 0 and y == 0:
        return np.nan
    ang = math.degrees(math.atan2(y, x))
    return ang % 360.0

def mean_direction_list(angles_deg):
    vals = [float(a) for a in angles_deg if not pd.isna(a)]
    if len(vals) == 0:
        return np.nan
    x = sum(math.cos(math.radians(a)) for a in vals)
    y = sum(math.sin(math.radians(a)) for a in vals)
    if x == 0 and y == 0:
        return np.nan
    ang = math.degrees(math.atan2(y, x))
    return ang % 360.0

required_cols = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]

def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    colmap = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ["est", "estacao", "estação"]:
            colmap[c] = "EST"
        elif low in ["pv", "ponto visado", "ponto_visado", "ponto"]:
            colmap[c] = "PV"
        elif "horizontal" in low and "pd" in low or ("hz" in low and "pd" in low):
            colmap[c] = "Hz_PD"
        elif "horizontal" in low and "pi" in low or ("hz" in low and "pi" in low):
            colmap[c] = "Hz_PI"
        elif "zenital" in low and "pd" in low or ("z" in low and "pd" in low):
            colmap[c] = "Z_PD"
        elif "zenital" in low and "pi" in low or ("z" in low and "pi" in low):
            colmap[c] = "Z_PI"
        elif "dist" in low and "pd" in low:
            colmap[c] = "DI_PD"
        elif "dist" in low and "pi" in low:
            colmap[c] = "DI_PI"
        else:
            colmap[c] = c
    df = df.rename(columns=colmap)
    return df

def validar_dataframe(df_original: pd.DataFrame):
    erros = []
    df = normalizar_colunas(df_original)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        erros.append("Colunas obrigatórias ausentes: " + ", ".join(missing))

    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    invalid_rows_hz, invalid_rows_z, invalid_rows_di = [], [], []
    for idx, row in df.iterrows():
        hz_pd = parse_angle_to_decimal(row.get("Hz_PD", ""))
        hz_pi = parse_angle_to_decimal(row.get("Hz_PI", ""))
        z_pd  = parse_angle_to_decimal(row.get("Z_PD", ""))
        z_pi  = parse_angle_to_decimal(row.get("Z_PI", ""))
        if pd.isna(hz_pd) or pd.isna(hz_pi):
            invalid_rows_hz.append(idx + 1)
        if pd.isna(z_pd) or pd.isna(z_pi):
            invalid_rows_z.append(idx + 1)
        try:
            di_pd = float(str(row.get("DI_PD", "")).replace(",", "."))
            di_pi = float(str(row.get("DI_PI", "")).replace(",", "."))
            if pd.isna(di_pd) or pd.isna(di_pi):
                invalid_rows_di.append(idx + 1)
        except Exception:
            invalid_rows_di.append(idx + 1)

    if invalid_rows_hz:
        erros.append(
            "Valores inválidos ou vazios em Hz_PD / Hz_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_hz)) + "."
        )
    if invalid_rows_z:
        erros.append(
            "Valores inválidos ou vazios em Z_PD / Z_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_z)) + "."
        )
    if invalid_rows_di:
        erros.append(
            "Valores inválidos ou vazios em DI_PD / DI_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_di)) + "."
        )

    return df, erros

# -------------------- Processamento principal ------------------------
df_uso = None

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)

        st.success(f"Arquivo '{uploaded.name}' carregado ({len(raw_df)} linhas).")

        df_valid, erros = validar_dataframe(raw_df)

        st.subheader("Pré-visualização dos dados importados")
        st.dataframe(df_valid[required_cols], use_container_width=True)

        if erros:
            st.error("Não foi possível calcular diretamente devido aos seguintes problemas:")
            for e in erros:
                st.markdown(f"- {e}")

            st.markdown("### Corrija os dados abaixo e clique em *Aplicar correções*")
            edited_df = st.data_editor(
                df_valid[required_cols],
                num_rows="dynamic",
                use_container_width=True,
                key="editor_corrigir_tudo",
            )

            if st.button("Aplicar correções"):
                df_corrigido, erros2 = validar_dataframe(edited_df)
                if not erros2:
                    st.success("Dados corrigidos com sucesso.")
                    df_uso = df_corrigido[required_cols].copy()
                else:
                    st.error("Ainda há problemas após a correção:")
                    for e in erros2:
                        st.markdown(f"- {e}")
        else:
            df_uso = df_valid[required_cols].copy()

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

# -------------------- Cálculos linha a linha + médias por par ------------------------
res = None          # linha a linha
df_par = None       # agregado por par EST–PV

if df_uso is not None:
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>3. Cálculos e resultados</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res = df_uso.copy()

    ref_por_estacao = {
        "P1": "P2",
        "P2": "P1",
        "P3": "P1",
    }

    def classificar_re_vante(est, pv):
        est_ = str(est).strip().upper()
        pv_  = str(pv).strip().upper()
        ref  = ref_por_estacao.get(est_)
        if ref is None:
            return ""
        return "Ré" if pv_ == ref else "Vante"

    res["Tipo"] = res.apply(lambda r: classificar_re_vante(r["EST"], r["PV"]), axis=1)

    # Ângulos em decimal (por linha)
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    # Distâncias inclinadas em float
    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    # DH/DN por linha
    z_pd_rad = res["Z_PD_deg"] * np.pi / 180.0
    z_pi_rad = res["Z_PI_deg"] * np.pi / 180.0

    res["DH_PD_m"] = np.abs(res["DI_PD_m"] * np.sin(z_pd_rad)).round(4)
    res["DN_PD_m"] = np.abs(res["DI_PD_m"] * np.cos(z_pd_rad)).round(4)
    res["DH_PI_m"] = np.abs(res["DI_PI_m"] * np.sin(z_pi_rad)).round(4)
    res["DN_PI_m"] = np.abs(res["DI_PI_m"] * np.cos(z_pi_rad)).round(4)

    res["Hz_med_deg"] = res.apply(
        lambda r: mean_direction_two(r["Hz_PD_deg"], r["Hz_PI_deg"]),
        axis=1
    )
    res["Hz_med_DMS"] = res["Hz_med_deg"].apply(decimal_to_dms)

    res["DH_med_m"] = np.abs((res["DH_PD_m"] + res["DH_PI_m"]) / 2.0).round(4)
    res["DN_med_m"] = np.abs((res["DN_PD_m"] + res["DN_PI_m"]) / 2.0).round(4)

    # ------- Agregação por par EST–PV -------
    def agg_par(df):
        out = {}
        out["Hz_PD_med_deg"] = mean_direction_list(df["Hz_PD_deg"])
        out["Hz_PI_med_deg"] = mean_direction_list(df["Hz_PI_deg"])
        out["Z_PD_med_deg"]  = mean_direction_list(df["Z_PD_deg"])
        out["Z_PI_med_deg"]  = mean_direction_list(df["Z_PI_deg"])
        out["DI_PD_med_m"]   = float(df["DI_PD_m"].mean())
        out["DI_PI_med_m"]   = float(df["DI_PI_m"].mean())
        return pd.Series(out)

    df_par = res.groupby(["EST", "PV"], as_index=False).apply(agg_par)

    df_par["Hz_med_deg_par"] = df_par.apply(
        lambda r: mean_direction_two(r["Hz_PD_med_deg"], r["Hz_PI_med_deg"]),
        axis=1
    )
    df_par["Hz_med_DMS_par"] = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    zpd_par_rad = df_par["Z_PD_med_deg"] * np.pi / 180.0
    zpi_par_rad = df_par["Z_PI_med_deg"] * np.pi / 180.0

    df_par["DH_PD_m_par"] = np.abs(df_par["DI_PD_med_m"] * np.sin(zpd_par_rad)).round(4)
    df_par["DN_PD_m_par"] = np.abs(df_par["DI_PD_med_m"] * np.cos(zpd_par_rad)).round(4)
    df_par["DH_PI_m_par"] = np.abs(df_par["DI_PI_med_m"] * np.sin(zpi_par_rad)).round(4)
    df_par["DN_PI_m_par"] = np.abs(df_par["DI_PI_med_m"] * np.cos(zpi_par_rad)).round(4)

    df_par["DH_med_m_par"] = np.abs(
        (df_par["DH_PD_m_par"] + df_par["DH_PI_m_par"]) / 2.0
    ).round(4)
    df_par["DN_med_m_par"] = np.abs(
        (df_par["DN_PD_m_par"] + df_par["DN_PI_m_par"]) / 2.0
    ).round(4)

    # ------- Tabelas -------
    resumo_df = pd.DataFrame({
        "EST": res["EST"],
        "PV": res["PV"],
        "Tipo": res["Tipo"],
        "Hz_PD": res["Hz_PD"],
        "Hz_PI": res["Hz_PI"],
        "Hz_médio (DMS)": res["Hz_med_DMS"].fillna(""),
        "DH_PD (m)": res["DH_PD_m"],
        "DH_PI (m)": res["DH_PI_m"],
        "DH_médio (m)": res["DH_med_m"],
        "DN_PD (m)": res["DN_PD_m"],
        "DN_PI (m)": res["DN_PI_m"],
        "DN_médio (m)": res["DN_med_m"],
    })

    st.markdown("##### Tabela linha a linha (cada leitura)")
    st.dataframe(resumo_df, use_container_width=True)

    resumo_par = pd.DataFrame({
        "EST": df_par["EST"],
        "PV": df_par["PV"],
        "Hz_PD_médio (deg)": df_par["Hz_PD_med_deg"].round(6),
        "Hz_PI_médio (deg)": df_par["Hz_PI_med_deg"].round(6),
        "Hz_médio (DMS)": df_par["Hz_med_DMS_par"],
        "DH_PD_médio (m)": df_par["DH_PD_m_par"],
        "DH_PI_médio (m)": df_par["DH_PI_m_par"],
        "DH_médio (m)": df_par["DH_med_m_par"],
        "DN_PD_médio (m)": df_par["DN_PD_m_par"],
        "DN_PI_médio (m)": df_par["DN_PI_m_par"],
        "DN_médio (m)": df_par["DN_med_m_par"],
    })

    st.markdown("##### Resultados médios por par EST–PV")
    st.dataframe(resumo_par, use_container_width=True)

    out_csv = resumo_par.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Baixar resultados médios (CSV)",
        data=out_csv,
        file_name="resultados_medios_estacao_total_ufpe.csv",
        mime="text/csv",
        key="download_saida_csv_par"
    )

# ==================== 4. Croqui gráfico e triângulo ====================
if df_par is not None and not df_par.empty:
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>4. Croqui gráfico e análise do triângulo P1–P2–P3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Para o <b>triângulo médio</b> P1–P2–P3 o programa usa as <b>médias de todas as leituras</b>
        de cada combinação EST–PV (PD e PI).<br>
        No modo de leituras específicas, você escolhe apenas um par EST⇒PV
        e o programa completa automaticamente o circuito P1⇒P3, P3⇒P2, P2⇒P1.
        """,
        unsafe_allow_html=True,
    )

    # ---------- 4.1 – Coordenadas aproximadas (df_par) ----------
    coords = {"P1": (0.0, 0.0)}

    def add_coord_from(est, pv, dh, hz_deg):
        est_ = str(est).strip().upper()
        pv_  = str(pv).strip().upper()
        if est_ not in coords:
            return
        x_est, y_est = coords[est_]
        az = math.radians(hz_deg)
        dx = dh * math.sin(az)
        dy = dh * math.cos(az)
        x_new = x_est + dx
        y_new = y_est + dy
        if pv_ in coords:
            x_old, y_old = coords[pv_]
            coords[pv_] = ((x_old + x_new) / 2.0, (y_old + y_new) / 2.0)
        else:
            coords[pv_] = (x_new, y_new)

    for _, row in df_par.iterrows():
        if str(row["EST"]).strip().upper() == "P1":
            add_coord_from(row["EST"], row["PV"], row["DH_med_m_par"], row["Hz_med_deg_par"])

    for _ in range(3):
        for _, row in df_par.iterrows():
            add_coord_from(row["EST"], row["PV"], row["DH_med_m_par"], row["Hz_med_deg_par"])

    pontos_basicos = {"P1", "P2", "P3"}
    if not pontos_basicos.issubset(set(coords.keys())):
        st.info(
            "Para montar o triângulo P1–P2–P3 é necessário ter médias para P1–P2, P1–P3 e P2–P3."
        )
    else:
        def dist(p, q, use_coords):
            x1, y1 = use_coords[p]
            x2, y2 = use_coords[q]
            return math.hypot(x2 - x1, y2 - y1)

        def angulo_oposto(lado_oposto, lado1, lado2):
            num = lado1**2 + lado2**2 - lado_oposto**2
            den = 2 * lado1 * lado2
            if den == 0:
                return np.nan
            cos_val = max(-1.0, min(1.0, num / den))
            return math.degrees(math.acos(cos_val))

        def info_triangulo(pA, pB, pC, use_coords):
            a = dist(pB, pC, use_coords)
            b = dist(pA, pC, use_coords)
            c = dist(pA, pB, use_coords)
            ang_A = angulo_oposto(a, b, c)
            ang_B = angulo_oposto(b, a, c)
            ang_C = angulo_oposto(c, a, b)
            s = (a + b + c) / 2.0
            area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
            lados = {
                "A": a, "B": b, "C": c,
                "nome_lado_A": f"{pB}{pC}",
                "nome_lado_B": f"{pA}{pC}",
                "nome_lado_C": f"{pA}{pB}",
            }
            angulos = {"A": ang_A, "B": ang_B, "C": ang_C}
            return lados, angulos, area

        def resumo_angulos(angA, angB, angC):
            soma = angA + angB + angC
            desvio = soma - 180.0
            return soma, desvio

        modo = st.radio(
            "Escolha o modo de construção do triângulo:",
            [
                "Triângulo médio P1–P2–P3 (todas as leituras)",
                "Triângulo a partir de leituras específicas (por par EST–PV)",
            ],
            index=0,
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        for nome, (x, y) in coords.items():
            cor = "darkred" if nome == "P1" else "navy"
            ax.scatter(x, y, color=cor, s=40, zorder=3)
            ax.text(x, y, f" {nome}", fontsize=9, va="bottom", ha="left")

        # ---------- MODO 1: triângulo médio ----------
        if modo.startswith("Triângulo médio"):
            pA, pB, pC = "P1", "P2", "P3"
            lados, angulos, area = info_triangulo(pA, pB, pC, coords)

            xA, yA = coords[pA]
            xB, yB = coords[pB]
            xC, yC = coords[pC]
            ax.plot([xA, xB], [yA, yB], "-k", linewidth=1.8, label="Triângulo médio")
            ax.plot([xB, xC], [yB, yC], "-k", linewidth=1.8)
            ax.plot([xA, xC], [yA, yC], "-k", linewidth=1.8)

            lados_df = pd.DataFrame({
                "Lado": [lados["nome_lado_C"], lados["nome_lado_A"], lados["nome_lado_B"]],
                "Distância (m)": [round(lados["C"], 4), round(lados["A"], 4), round(lados["B"], 4)],
            })
            ang_df = pd.DataFrame({
                "Vértice": [pA, pB, pC],
                "Ângulo interno (graus)": [
                    round(angulos["A"], 4),
                    round(angulos["B"], 4),
                    round(angulos["C"], 4),
                ],
            })

            st.markdown("#### Triângulo médio P1–P2–P3 (todas as leituras)")
            st.markdown("##### Distâncias dos lados")
            st.dataframe(lados_df, use_container_width=True)

            st.markdown("##### Ângulos internos (lei dos cossenos)")
            st.dataframe(ang_df, use_container_width=True)

            soma_ang, desvio = resumo_angulos(
                angulos["A"], angulos["B"], angulos["C"]
            )
            st.markdown(
                f"**Soma dos ângulos internos:** `{soma_ang:.4f}°` &nbsp;&nbsp; "
                f"(desvio em relação a 180°: `{desvio:+.4f}°`)"
            )

            st.markdown(f"**Área do triângulo (Heron):** `{area:.4f} m²`")

        # ---------- MODO 2: específico com anotação de lados/ângulos ----------
        else:
            st.markdown(
                """
                Escolha uma combinação <b>Estação (EST) – Ponto de Visada (PV)</b>.
                O programa seleciona automaticamente as outras duas estações
                necessárias para fechar o triângulo P1–P2–P3:
                <ul>
                  <li>Se você escolher <b>P1 ⇒ P3</b>, o programa usará também <b>P3 ⇒ P2</b> e <b>P2 ⇒ P1</b>;</li>
                  <li>Se você escolher <b>P3 ⇒ P2</b>, o programa usará também <b>P2 ⇒ P1</b> e <b>P1 ⇒ P3</b>;</li>
                  <li>Se você escolher <b>P2 ⇒ P1</b>, o programa usará também <b>P1 ⇒ P3</b> e <b>P3 ⇒ P2</b>.</li>
                </ul>
                O triângulo é sempre P1–P3–P2, partindo de P1 como referência.
                """,
                unsafe_allow_html=True,
            )

            df3 = df_par[
                df_par["EST"].isin(["P1", "P2", "P3"]) &
                df_par["PV"].isin(["P1", "P2", "P3"])
            ].copy()

            if df3.empty:
                st.info("Não há médias entre P1, P2 e P3 suficientes para este modo.")
            else:
                df3["par"] = df3["EST"].astype(str).str.upper() + "⇒" + df3["PV"].astype(str).str.upper()
                pares_disponiveis = sorted(df3["par"].unique())
                pares_validos = [p for p in pares_disponiveis if p in ["P1⇒P3", "P3⇒P2", "P2⇒P1"]]

                if not pares_validos:
                    st.warning("Não foram encontradas médias P1⇒P3, P3⇒P2 ou P2⇒P1.")
                else:
                    par_escolhido = st.selectbox(
                        "Escolha um par Estação ⇒ PV para iniciar o triângulo:",
                        options=pares_validos,
                        index=pares_validos.index("P1⇒P3") if "P1⇒P3" in pares_validos else 0,
                    )

                    # circuito sempre P1→P3, P3→P2, P2→P1
                    par_13 = "P1⇒P3"
                    par_32 = "P3⇒P2"
                    par_21 = "P2⇒P1"

                    def dh_med_par(par_str):
                        sub = df3[df3["par"] == par_str]
                        if sub.empty:
                            return np.nan
                        return float(sub["DH_med_m_par"].iloc[0])

                    L13 = dh_med_par(par_13)  # P1–P3
                    L32 = dh_med_par(par_32)  # P3–P2
                    L21 = dh_med_par(par_21)  # P2–P1

                    if any(pd.isna(v) or v == 0 for v in [L13, L32, L21]):
                        st.warning(
                            "Não foi possível calcular todas as distâncias médias. "
                            "Verifique se existem médias para P1⇒P3, P3⇒P2 e P2⇒P1."
                        )
                    else:
                        # Triângulo abstrato P1–P3–P2 partindo de P1
                        a = L32  # lado oposto a P1
                        b = L21  # lado oposto a P3
                        c = L13  # lado oposto a P2

                        coords_tri = {
                            "P1": (0.0, 0.0),
                            "P3": (c, 0.0),
                        }

                        cos_A = max(-1.0, min(1.0, (b**2 + c**2 - a**2) / (2 * b * c)))
                        ang_A_rad = math.acos(cos_A)
                        x2 = b * math.cos(ang_A_rad)
                        y2 = b * math.sin(ang_A_rad)
                        coords_tri["P2"] = (x2, y2)

                        def dist_t(p, q):
                            x1, y1 = coords_tri[p]
                            x2_, y2_ = coords_tri[q]
                            return math.hypot(x2_ - x1, y2_ - y1)

                        d_P1P3 = dist_t("P1", "P3")
                        d_P3P2 = dist_t("P3", "P2")
                        d_P2P1 = dist_t("P2", "P1")

                        ang_P1 = angulo_oposto(d_P3P2, d_P1P3, d_P2P1)
                        ang_P3 = angulo_oposto(d_P2P1, d_P1P3, d_P3P2)
                        ang_P2 = angulo_oposto(d_P1P3, d_P2P1, d_P3P2)

                        soma_ang, desvio = resumo_angulos(ang_P1, ang_P3, ang_P2)

                        s_ = (d_P1P3 + d_P3P2 + d_P2P1) / 2.0
                        area_ = math.sqrt(max(0.0, s_ * (s_ - d_P1P3) * (s_ - d_P3P2) * (s_ - d_P2P1)))

                        # Desenho com distâncias e ângulos anotados
                        x1, y1 = coords_tri["P1"]
                        x3, y3 = coords_tri["P3"]
                        x2, y2 = coords_tri["P2"]

                        ax.plot([x1, x3], [y1, y3], "-k", linewidth=1.8, label="Triângulo escolhido")
                        ax.plot([x3, x2], [y3, y2], "-k", linewidth=1.8)
                        ax.plot([x2, x1], [y2, y1], "-k", linewidth=1.8)

                        for nome, (x, y) in coords_tri.items():
                            cor = {"P1": "red", "P2": "blue", "P3": "green"}.get(nome, "navy")
                            ax.scatter(x, y, color=cor, s=55, zorder=4)
                            ax.text(x, y, f" {nome}", fontsize=10, va="bottom", ha="left")

                        def meio(Pa, Pb):
                            xA, yA = coords_tri[Pa]
                            xB, yB = coords_tri[Pb]
                            return (xA + xB) / 2.0, (yA + yB) / 2.0

                        mx_13, my_13 = meio("P1", "P3")
                        mx_32, my_32 = meio("P3", "P2")
                        mx_21, my_21 = meio("P2", "P1")

                        ax.text(
                            mx_13, my_13,
                            f"P1–P3 = {d_P1P3:.4f} m",
                            fontsize=8, color="black", ha="center", va="bottom"
                        )
                        ax.text(
                            mx_32, my_32,
                            f"P3–P2 = {d_P3P2:.4f} m",
                            fontsize=8, color="black", ha="center", va="bottom"
                        )
                        ax.text(
                            mx_21, my_21,
                            f"P2–P1 = {d_P2P1:.4f} m",
                            fontsize=8, color="black", ha="center", va="bottom"
                        )

                        ax.text(
                            x1, y1,
                            f" ∠P1 = {ang_P1:.4f}°",
                            fontsize=8, color="darkred", ha="left", va="top"
                        )
                        ax.text(
                            x3, y3,
                            f" ∠P3 = {ang_P3:.4f}°",
                            fontsize=8, color="darkgreen", ha="right", va="bottom"
                        )
                        ax.text(
                            x2, y2,
                            f" ∠P2 = {ang_P2:.4f}°",
                            fontsize=8, color="darkblue", ha="left", va="bottom"
                        )

                        st.markdown("#### Triângulo (médias P1⇒P3, P3⇒P2, P2⇒P1) com distâncias e ângulos")

                        lados_df = pd.DataFrame({
                            "Lado": ["P1–P3", "P3–P2", "P2–P1"],
                            "Distância (m)": [
                                round(d_P1P3, 4),
                                round(d_P3P2, 4),
                                round(d_P2P1, 4),
                            ],
                        })
                        ang_df = pd.DataFrame({
                            "Vértice": ["P1", "P3", "P2"],
                            "Ângulo interno (graus)": [
                                round(ang_P1, 4),
                                round(ang_P3, 4),
                                round(ang_P2, 4),
                            ],
                        })

                        st.markdown("##### Distâncias dos lados")
                        st.dataframe(lados_df, use_container_width=True)

                        st.markdown("##### Ângulos internos (lei dos cossenos)")
                        st.dataframe(ang_df, use_container_width=True)

                        st.markdown(
                            f"**Soma dos ângulos internos:** `{soma_ang:.4f}°` &nbsp;&nbsp; "
                            f"(desvio em relação a 180°: `{desvio:+.4f}°`)"
                        )
                        st.markdown(f"**Área do triângulo (Heron):** `{area_:.4f} m²`")

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X local (m)")
        ax.set_ylabel("Y local (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        st.pyplot(fig)

# -------------------- Rodapé -------------------------
st.markdown(
    """
    <p class="footer-text">
        Versão do app: <code>6.1 — Médias por par EST–PV + Triângulo anotado (lados, ângulos, desvio)</code>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # fim main-card
