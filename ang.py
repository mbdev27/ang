# app.py
# Média das Direções (Hz) + DH/DN - UFPE
# Rode com: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math

# -------------------- Configuração da página --------------------
st.set_page_config(
    page_title="Média das Direções (Hz) — Estação Total | UFPE",
    layout="wide",
    page_icon="📐"
)

# -------------------- Estilos customizados (cores UFPE + degradês) --------------------
CUSTOM_CSS = """
<style>
/* Fundo geral com leve degradê, independente do modo claro/escuro do navegador */
body, .stApp {
    background: radial-gradient(circle at top left, #faf5f5 0%, #f7f5f5 45%, #f4f4f4 100%);
    color: #111827;
    font-family: "Trebuchet MS", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Container principal em forma de cartão */
.main-card {
    background: linear-gradient(145deg, #ffffff 0%, #fdfbfb 40%, #ffffff 100%);
    border-radius: 18px;
    padding: 1.8rem 2.1rem;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 18px 40px rgba(15, 23, 42, 0.22),
        0 0 0 1px rgba(15, 23, 42, 0.03);
}

/* Faixa superior bordô com leve brilho */
.ufpe-top-bar {
    width: 100%;
    height: 8px;
    border-radius: 0 0 14px 14px;
    background: linear-gradient(90deg, #5b0000 0%, #990000 52%, #5b0000 100%);
    margin-bottom: 0.9rem;
}

/* Header UFPE */
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

/* Título do app */
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

/* Subtítulo */
.app-subtitle {
    font-size: 0.95rem;
    color: #4b5563;
    margin-bottom: 0.9rem;
}

/* Section titles */
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

/* Caixa de ajuda */
.helper-box {
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    background: linear-gradient(135deg, #fdf2f2 0%, #fff5f5 40%, #fdf2f2 100%);
    border: 1px solid rgba(153, 0, 0, 0.25);
    font-size: 0.83rem;
    color: #4b5563;
    margin-bottom: 0.7rem;
}

/* Rodapé */
.footer-text {
    font-size: 0.75rem;
    color: #6b7280;
}

/* Download buttons */
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

/* Tabelas e editores em cartões */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background: linear-gradient(145deg, #ffffff 0%, #f9fafb 60%, #ffffff 100%) !important;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
}

/* Cabeçalho das tabelas */
[data-testid="stDataFrame"] thead tr {
    background: linear-gradient(90deg, #f5e6e8 0%, #fdf2f2 100%) !important;
    color: #5b101d !important;
    font-weight: 600;
}

/* Linhas alternadas */
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: #fafafa !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #f3f0f0 !important;
}

/* Código da saída (se usado) */
[data-testid="stCodeBlock"] {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #f9fafb !important;
}

/* Labels dos inputs */
.stTextInput label, .stFileUploader label {
    font-size: 0.86rem;
    font-weight: 600;
    color: #374151;
}

/* Inputs com leve degradê e bordas suaves */
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

/* Botões "pill" com degradê */
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

/* Expander header */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #7d1220 !important;
}

/* Alertas arredondados */
.stAlert {
    border-radius: 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Cabeçalho ----------------------
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="ufpe-top-bar"></div>', unsafe_allow_html=True)

    # ---- Bloco UFPE + informações do curso ----
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

        # Campos vazios e preenchíveis
        col_prof, col_local, col_equip, col_data, col_patr = st.columns(
            [1.6, 1.4, 1.6, 1.1, 1.2]
        )
        professor = col_prof.text_input("Professor", value="")
        local = col_local.text_input("Local", value="")
        equipamento = col_equip.text_input("Equipamento", value="")
        data_trabalho = col_data.text_input("Data", value="")
        patrimonio = col_patr.text_input("Patrimônio", value="")

    st.markdown('<hr class="ufpe-separator">', unsafe_allow_html=True)

    # ---- Título do app ----
    st.markdown(
        """
        <div class="app-title">
            <span class="icon">📐</span>
            <span>Média das Direções (Hz) — Estação Total</span>
        </div>
        <div class="app-subtitle">
            Cálculo da média das direções Hz e das distâncias horizontais/níveis
            a partir de leituras PD e PI, com validação automática da planilha de campo.
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
            Distâncias inclinadas em <b>metros</b> (ex.: 25.365).
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Template Excel para download -------------------------
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

# -------------------- Upload de arquivo -------------------------
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

# -------------------- Funções auxiliares de ângulo --------------------------
angle_re = re.compile(r"(-?\d+)[^\d\-]+(\d+)[^\d\-]+(\d+(?:[.,]\d+)?)")
num_re = re.compile(r"^-?\d+(?:[.,]\d+)?$")

def parse_angle_to_decimal(x):
    """Aceita DMS (ex: 89°48'20\" ou 89 48 20 ou 89:48:20) ou decimal (89.8056)."""
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

    # Tenta DMS explícito
    m = angle_re.search(s)
    if m:
        deg = float(m.group(1))
        minu = float(m.group(2))
        sec = float(m.group(3).replace(",", "."))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu/60.0 + sec/3600.0)

    # Decimal puro
    if num_re.match(s.replace(" ", "")):
        return float(s.replace(",", "."))

    # D M S separados
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

def mean_direction_deg(a_deg, b_deg):
    """Média vetorial de duas direções em graus."""
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

# -------------------- Validação do DataFrame ------------------------
required_cols = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]

def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
    """Permite alguns nomes próximos e normaliza para o padrão exigido."""
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
        erros.append(
            "Colunas obrigatórias ausentes: " + ", ".join(missing)
        )

    # Garante colunas (vazias) para edição, mesmo se faltarem
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    # Checa ângulos
    invalid_rows_hz = []
    invalid_rows_z = []
    for idx, row in df.iterrows():
        hz_pd = parse_angle_to_decimal(row.get("Hz_PD", ""))
        hz_pi = parse_angle_to_decimal(row.get("Hz_PI", ""))
        z_pd = parse_angle_to_decimal(row.get("Z_PD", ""))
        z_pi = parse_angle_to_decimal(row.get("Z_PI", ""))
        if pd.isna(hz_pd) or pd.isna(hz_pi):
            invalid_rows_hz.append(idx + 1)
        if pd.isna(z_pd) or pd.isna(z_pi):
            invalid_rows_z.append(idx + 1)
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

    # Checa distâncias
    invalid_rows_di = []
    for idx, row in df.iterrows():
        try:
            di_pd = float(str(row.get("DI_PD", "")).replace(",", "."))
            di_pi = float(str(row.get("DI_PI", "")).replace(",", "."))
        except Exception:
            invalid_rows_di.append(idx + 1)
            continue
        if pd.isna(di_pd) or pd.isna(di_pi):
            invalid_rows_di.append(idx + 1)
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
            st.error(
                "Não foi possível calcular diretamente devido aos seguintes problemas:"
            )
            for e in erros:
                st.markdown(f"- {e}")

            st.markdown(
                "### Corrija os dados abaixo e clique em *Aplicar correções*"
            )
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

# -------------------- Cálculos Hz, DH, DN ------------------------
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

    # Marca Ré/Vante: aqui estou considerando que PV = P1 é Ré
    res["Tipo"] = res["PV"].apply(lambda pv: "Ré" if str(pv).strip().upper() == "P1" else "Vante")

    # Converte ângulos para graus decimais
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    # Hz médio (média vetorial)
    res["Hz_med_deg"] = res.apply(
        lambda r: mean_direction_deg(r["Hz_PD_deg"], r["Hz_PI_deg"]),
        axis=1
    )
    res["Hz_med_DMS"] = res["Hz_med_deg"].apply(decimal_to_dms)

    # Distâncias inclinadas
    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    # Z -> radianos (zenital)
    z_pd_rad = res["Z_PD_deg"] * np.pi / 180.0
    z_pi_rad = res["Z_PI_deg"] * np.pi / 180.0

    # Distância horizontal e diferença de nível
    res["DH_PD_m"] = (res["DI_PD_m"] * np.sin(z_pd_rad)).round(3)
    res["DN_PD_m"] = (res["DI_PD_m"] * np.cos(z_pd_rad)).round(3)
    res["DH_PI_m"] = (res["DI_PI_m"] * np.sin(z_pi_rad)).round(3)
    res["DN_PI_m"] = (res["DI_PI_m"] * np.cos(z_pi_rad)).round(3)

    res["DH_med_m"] = ((res["DH_PD_m"] + res["DH_PI_m"]) / 2.0).round(3)
    res["DN_med_m"] = ((res["DN_PD_m"] + res["DN_PI_m"]) / 2.0).round(3)

    # ----------------- Tabela resumo (Hz e distâncias) -----------------
    resumo_df = pd.DataFrame({
        "EST": res["EST"],
        "PV": res["PV"],
        "Tipo": res["Tipo"],                 # <<< Ré / Vante
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

    st.dataframe(resumo_df, use_container_width=True)

    # ----------------- Download da saída -------------------------
    out_df = resumo_df.copy()
    out_csv = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Baixar resultados (CSV)",
        data=out_csv,
        file_name="resultados_estacao_total_ufpe.csv",
        mime="text/csv",
        key="download_saida_csv"
    )

# -------------------- Rodapé -------------------------
st.markdown(
    """
    <p class="footer-text">
        Versão do app: <code>3.1 — Hz + DH/DN (modelo UFPE, degradês, Ré/Vante)</code>.<br>
        Para gerar/baixar o modelo Excel (.xlsx) no servidor,
        certifique-se de incluir <code>openpyxl</code> no <code>requirements.txt</code>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # fim main-card
