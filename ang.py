# app.py
# Média das Direções (Hz) - UFPE
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

# -------------------- Estilos customizados (cores UFPE) --------------------
CUSTOM_CSS = """
<style>
/* Fundo geral em tom claro com faixa bordô no topo */
.stApp {
    background: #f7f5f5;
    color: #111827;
    font-family: "Trebuchet MS", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Container principal */
.main-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.8rem 2.1rem;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 14px 30px rgba(15, 23, 42, 0.25),
        0 0 0 1px rgba(15, 23, 42, 0.04);
}

/* Faixa superior bordô */
.ufpe-top-bar {
    width: 100%;
    height: 8px;
    border-radius: 0 0 12px 12px;
    background: #990000;
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
    background: linear-gradient(135deg, #990000, #111827);
}

/* Caixa de ajuda */
.helper-box {
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    background: #fdf2f2;
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
    background: #990000;
    color: white;
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.45rem 0.95rem;
}
.stDownloadButton > button:hover {
    border-color: #111827;
    background: #111827;
}

/* Tabelas e editores */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background-color: #ffffff !important;
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
            Cálculo da média das direções Hz a partir de leituras PD e PI, com validação
            automática da planilha de campo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="helper-box">
            <b>Entrada mínima:</b> colunas <code>EST</code>, <code>PV</code>, <code>Hz_PD</code> e <code>Hz_PI</code>.<br>
            Os ângulos podem ser em <b>DMS</b> (ex.: 359°59'54") ou <b>decimal</b> (ex.: 359.9983).<br>
            Em caso de dados faltando ou inválidos, o sistema indicará os problemas e
            abrirá uma tabela para correção manual.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Template Excel para download -------------------------
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>1. Modelo de dados (mínimo para cálculo)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_df = pd.DataFrame({
        'EST': ['A', 'A'],
        'PV': ['B', 'C'],
        'Hz_PD': ["00°00'00\"", "18°58'22\""],
        'Hz_PI': ["179°59'48\"", "198°58'14\""],
    })

    excel_bytes = io.BytesIO()
    try:
        template_df.to_excel(excel_bytes, index=False)
        excel_bytes.seek(0)
        st.download_button(
            "📥 Baixar modelo Excel (.xlsx)",
            data=excel_bytes.getvalue(),
            file_name="modelo_media_direcoes_hz.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_model"
        )
    except Exception:
        csv_bytes = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar modelo CSV",
            data=csv_bytes,
            file_name="modelo_media_direcoes_hz.csv",
            mime="text/csv",
            key="download_csv_model"
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
        "Envie a planilha preenchida (Hz_PD / Hz_PI)",
        type=["xlsx", "xls", "csv"],
        help="Se a planilha estiver incompleta, o sistema mostrará quais campos corrigir."
    )

# -------------------- Funções auxiliares --------------------------
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
    # normaliza para [0, 360)
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
    """
    Média de duas direções em graus, respeitando o círculo (0-360).
    Usamos média vetorial (cos/sin) para tratar casos em torno de 0°/360°.
    """
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

# -------------------- Processamento / Validação ------------------------
required_min_cols = ["EST", "PV", "Hz_PD", "Hz_PI"]

def validar_dataframe(df_original: pd.DataFrame):
    """
    Verifica colunas mínimas e valores de ângulo.
    Retorna (df_corrigido, lista_erros, df_exemplo).
    """
    erros = []

    df = df_original.copy()
    # normaliza nomes possíveis para as mínimas
    colmap = {c: c for c in df.columns}
    for c in list(df.columns):
        low = c.strip().lower()
        if low in ["est", "estacao", "estação"]:
            colmap[c] = "EST"
        if low in ["pv", "ponto visado", "ponto_visado", "ponto"]:
            colmap[c] = "PV"
        if "hz" in low and "pd" in low:
            colmap[c] = "Hz_PD"
        if "hz" in low and "pi" in low:
            colmap[c] = "Hz_PI"

    df = df.rename(columns=colmap)

    missing_cols = [c for c in required_min_cols if c not in df.columns]
    if missing_cols:
        erros.append(
            f"Colunas obrigatórias ausentes: {', '.join(missing_cols)}."
        )

    # Garante colunas mínimas (mesmo que vazias) para permitir edição
    for c in required_min_cols:
        if c not in df.columns:
            df[c] = ""

    # Validação de ângulos Hz_PD / Hz_PI
    invalid_rows = []
    for idx, row in df.iterrows():
        hz_pd_raw = row.get("Hz_PD", "")
        hz_pi_raw = row.get("Hz_PI", "")
        hz_pd_deg = parse_angle_to_decimal(hz_pd_raw)
        hz_pi_deg = parse_angle_to_decimal(hz_pi_raw)
        if pd.isna(hz_pd_deg) or pd.isna(hz_pi_deg):
            invalid_rows.append(idx + 1)  # linha 1-based
    if invalid_rows:
        erros.append(
            "Valores inválidos ou vazios em Hz_PD / Hz_PI nas linhas: "
            + ", ".join(map(str, invalid_rows))
            + "."
        )

    # Exemplo didático para mostrar em caso de erro
    exemplo_df = pd.DataFrame({
        "EST": ["A", "A"],
        "PV": ["B", "C"],
        "Hz_PD": ["00°00'00\"", "18°59'34\""],
        "Hz_PI": ["179°59'48\"", "198°59'24\""],
    })

    return df, erros, exemplo_df

# -------------------- Execução principal ------------------------
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)

        st.success(f"Arquivo '{uploaded.name}' carregado ({len(raw_df)} linhas).")

        df_valid, erros, exemplo = validar_dataframe(raw_df)

        if erros:
            st.error(
                "Não foi possível calcular a média das direções (Hz) "
                "devido aos seguintes problemas:"
            )
            for e in erros:
                st.markdown(f"- {e}")

            st.markdown("**Exemplo mínimo de preenchimento válido:**")
            st.dataframe(exemplo, use_container_width=True)

            st.markdown(
                "### Corrija/complete os dados abaixo e clique em *Aplicar* para recalcular"
            )
            edited_df = st.data_editor(
                df_valid[required_min_cols],
                num_rows="dynamic",
                use_container_width=True,
                key="editor_corrigir_hz",
            )

            # Revalida após edição
            df_valid2, erros2, _ = validar_dataframe(edited_df)
            if not erros2:
                df_em_uso = df_valid2
                st.success("Dados corrigidos. Cálculo da média de Hz realizado abaixo.")
            else:
                st.warning(
                    "Ainda há problemas nos dados após a edição. "
                    "Revise os campos destacados na mensagem acima."
                )
                df_em_uso = None
        else:
            df_em_uso = df_valid[required_min_cols].copy()

        # Se temos um dataframe válido, prossegue com os cálculos
        if df_em_uso is not None:
            st.markdown(
                """
                <div class="section-title">
                    <span class="dot"></span>
                    <span>3. Cálculos e resultados</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            results = df_em_uso.copy()

            # Converte Hz_PD / Hz_PI para graus decimais
            results["Hz_PD_deg"] = results["Hz_PD"].apply(parse_angle_to_decimal)
            results["Hz_PI_deg"] = results["Hz_PI"].apply(parse_angle_to_decimal)

            # Média das direções usando média vetorial
            results["Hz_med_deg"] = results.apply(
                lambda r: mean_direction_deg(r["Hz_PD_deg"], r["Hz_PI_deg"]),
                axis=1
            )
            results["Hz_med_DMS"] = results["Hz_med_deg"].apply(decimal_to_dms)

            # ----------------- Tabela de resultados (resumo) -----------------
            resumo_df = pd.DataFrame({
                "EST": results["EST"],
                "PV": results["PV"],
                "Hz_PD": results["Hz_PD"],
                "Hz_PI": results["Hz_PI"],
                "Hz_médio (DMS)": results["Hz_med_DMS"].fillna(""),
            })

            st.dataframe(resumo_df, use_container_width=True)

            # ----------------- Tabela de conferência -------------------------
            display_df = pd.DataFrame({
                "EST": results["EST"],
                "PV": results["PV"],
                "Ângulo Horizontal (PD)": results["Hz_PD"],
                "Ângulo Horizontal (PI)": results["Hz_PI"],
                "Hz Médio (DMS)": results["Hz_med_DMS"].fillna(""),
            })

            st.markdown(
                """
                <div class="section-title">
                    <span class="dot"></span>
                    <span>Tabela de conferência (valores angulares)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(display_df, use_container_width=True)

            # ----------------- Download da saída -------------------------
            out_df = results[["EST", "PV", "Hz_PD", "Hz_PI", "Hz_med_DMS"]].copy()
            out_df.rename(
                columns={
                    "Hz_PD": "Hz_PD (entrada)",
                    "Hz_PI": "Hz_PI (entrada)",
                    "Hz_med_DMS": "Hz_médio (DMS)",
                },
                inplace=True
            )
            out_csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar saída (CSV)",
                data=out_csv,
                file_name="saida_media_direcoes_hz.csv",
                mime="text/csv",
                key="download_saida_csv"
            )

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

# -------------------- Rodapé -------------------------
st.markdown(
    """
    <p class="footer-text">
        Observação: para gerar/baixar o modelo Excel (.xlsx) no servidor,
        certifique-se de incluir <code>openpyxl</code> no <code>requirements.txt</code>.<br>
        Versão do app: <code>2.0 — Média Hz (cores UFPE, validação de planilha)</code>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # fim main-card
