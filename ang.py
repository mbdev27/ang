# app.py
# Calculadora de Ângulos e Distâncias (Estação Total)
# Rode com: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math

# -------------------- Configuração da página --------------------
st.set_page_config(
    page_title="Calculadora Topográfica — Estação Total",
    layout="wide",
    page_icon="📐"
)

# -------------------- Estilos customizados ----------------------
CUSTOM_CSS = """
<style>
/* Fundo geral */
.stApp {
    background: radial-gradient(circle at top left, #1e293b 0, #020617 55%, #000 100%);
    color: #e5e7eb;
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Container principal */
.main-card {
    background: rgba(15, 23, 42, 0.90);
    border-radius: 18px;
    padding: 1.8rem 2.1rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow:
        0 24px 60px rgba(15, 23, 42, 0.9),
        0 0 0 1px rgba(15, 23, 42, 0.9);
}

/* Header UFPE */
.ufpe-header-text {
    font-size: 0.78rem;
    line-height: 1.15rem;
    text-transform: uppercase;
}
.ufpe-header-text strong {
    letter-spacing: 0.03em;
}
.ufpe-separator {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.45);
    margin: 0.8rem 0 1.0rem 0;
}

/* Título do app */
.app-title {
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.35rem;
}
.app-title span.icon {
    font-size: 2.6rem;
}

/* Subtítulo */
.app-subtitle {
    font-size: 0.95rem;
    color: #9ca3af;
    margin-bottom: 0.9rem;
}

/* Section titles */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.section-title span.dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(135deg, #38bdf8, #a855f7);
}

/* Caixa de ajuda */
.helper-box {
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(148, 163, 184, 0.45);
    font-size: 0.83rem;
    color: #d1d5db;
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
    border: 1px solid rgba(56, 189, 248, 0.8);
    background: radial-gradient(circle at top left, #0ea5e9 0, #0369a1 40%, #0f172a 100%);
    color: white;
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.45rem 0.95rem;
}
.stDownloadButton > button:hover {
    border-color: rgba(129, 140, 248, 0.9);
    background: radial-gradient(circle at top left, #38bdf8 0, #4f46e5 40%, #020617 100%);
}

/* Tabela de dados */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background-color: rgba(15, 23, 42, 0.92) !important;
}

/* Código da saída */
[data-testid="stCodeBlock"] {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.6);
    background: radial-gradient(circle at top left, #020617 0, #020617 45%, #020617 100%) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Cabeçalho ----------------------
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

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

    # ---- Título original do app ----
    st.markdown(
        """
        <div class="app-title">
            <span class="icon">📐</span>
            <span>Calculadora de Ângulos e Distâncias — Estação Total</span>
        </div>
        <div class="app-subtitle">
            Ferramenta para processar leituras PD / PI de Estação Total e gerar
            distâncias horizontais e ângulo horizontal médio.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="helper-box">
            <b>Entrada:</b> pares <code>PD / PI</code> de <b>Ângulo Horizontal</b>, <b>Ângulo Zenital</b> e
            <b>Distância Inclinada</b> por estação.<br>
            <b>Como usar:</b>
            <ul style="margin-top: 0.25rem; margin-bottom: 0; padding-left: 1.2rem;">
                <li>Baixe o modelo Excel, preencha e faça upload.</li>
                <li>Os ângulos podem ser em <b>DMS</b> (ex.: 235°47'33") ou <b>decimal</b> (ex.: 235.7925).</li>
                <li>A saída final mostra: <code>Dh_PD (m)</code> · <code>Dh_PI (m)</code> · <code>AH_médio (DMS)</code>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Template Excel para download -------------------------
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>1. Modelo de dados</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_df = pd.DataFrame({
        'EST': ['P1'],
        'PV': ['P2'],
        'AnguloHorizontal_PD': [''],
        'AnguloHorizontal_PI': [''],
        'AnguloZenital_PD': [''],
        'AnguloZenital_PI': [''],
        'DistanciaInclinada_PD': [''],
        'DistanciaInclinada_PI': ['']
    })

    excel_bytes = io.BytesIO()
    try:
        template_df.to_excel(excel_bytes, index=False)
        excel_bytes.seek(0)
        st.download_button(
            "📥 Baixar modelo Excel (.xlsx)",
            data=excel_bytes.getvalue(),
            file_name="modelo_estacao_total.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_model"
        )
    except Exception:
        csv_bytes = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar modelo CSV",
            data=csv_bytes,
            file_name="modelo_estacao_total.csv",
            mime="text/csv",
            key="download_csv_model"
        )

# -------------------- Upload de arquivo -------------------------
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>2. Carregar dados</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Envie a planilha preenchida",
        type=["xlsx", "xls", "csv"],
        help="O app usará os dados desta planilha para todos os cálculos."
    )

# -------------------- Regex / Funções auxiliares --------------------------
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
    sign = "-" if angle < 0 else ""
    a = abs(float(angle))
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
    return f"{sign}{g}°{m:02d}'{s:02d}\""


# -------------------- Preparar dataframe e cálculos ------------------------
required_cols = [
    'EST', 'PV',
    'AnguloHorizontal_PD', 'AnguloHorizontal_PI',
    'AnguloZenital_PD', 'AnguloZenital_PI',
    'DistanciaInclinada_PD', 'DistanciaInclinada_PI'
]

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success(f"Arquivo '{uploaded.name}' carregado com sucesso ({len(df)} linhas).")

        # mapeamento de colunas
        new_df_columns = {c: c for c in df.columns}
        for c in list(df.columns):
            low = c.strip().lower()
            if "horizontal" in low or ("ang" in low and "h" in low):
                if "pd" in low:
                    new_df_columns[c] = "AnguloHorizontal_PD"
                elif "pi" in low:
                    new_df_columns[c] = "AnguloHorizontal_PI"
            if "zenit" in low or "zenital" in low:
                if "pd" in low:
                    new_df_columns[c] = "AnguloZenital_PD"
                elif "pi" in low:
                    new_df_columns[c] = "AnguloZenital_PI"
            if "dist" in low and ("di" in low or "inclin" in low):
                if "pd" in low:
                    new_df_columns[c] = "DistanciaInclinada_PD"
                elif "pi" in low:
                    new_df_columns[c] = "DistanciaInclinada_PI"

        df = df.rename(columns=new_df_columns)
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        st.markdown(
            """
            <div class="section-title">
                <span class="dot"></span>
                <span>3. Cálculos e resultados</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        results = df[required_cols].copy()

        for c in ['AnguloHorizontal_PD', 'AnguloHorizontal_PI',
                  'AnguloZenital_PD', 'AnguloZenital_PI']:
            results[c + '_deg'] = results[c].apply(parse_angle_to_decimal)

        for c in ['DistanciaInclinada_PD', 'DistanciaInclinada_PI']:
            results[c + '_m'] = pd.to_numeric(results[c], errors='coerce')

        results['Dh_PD_m'] = results.apply(
            lambda r: (
                r['DistanciaInclinada_PD_m'] *
                math.sin(math.radians(r['AnguloZenital_PD_deg']))
            )
            if pd.notna(r.get('DistanciaInclinada_PD_m'))
            and pd.notna(r.get('AnguloZenital_PD_deg'))
            else np.nan,
            axis=1
        )

        results['Dh_PI_m'] = results.apply(
            lambda r: (
                r['DistanciaInclinada_PI_m'] *
                math.sin(math.radians(r['AnguloZenital_PI_deg']))
            )
            if pd.notna(r.get('DistanciaInclinada_PI_m'))
            and pd.notna(r.get('AnguloZenital_PI_deg'))
            else np.nan,
            axis=1
        )

        results['AH_med_deg'] = results[
            ['AnguloHorizontal_PD_deg', 'AnguloHorizontal_PI_deg']
        ].mean(axis=1, skipna=True)
        results['AH_med_DMS'] = results['AH_med_deg'].apply(decimal_to_dms)

        # -------------------- Saída em linhas --------------------
        lines = []
        for _, r in results.iterrows():
            dh_pd = r.get('Dh_PD_m')
            dh_pi = r.get('Dh_PI_m')
            ah_dms = r.get('AH_med_DMS') or ""
            dh_pd_s = f"{dh_pd:.3f} m" if pd.notna(dh_pd) else ""
            dh_pi_s = f"{dh_pi:.3f} m" if pd.notna(dh_pi) else ""
            lines.append(f"{dh_pd_s}\t{dh_pi_s}\t{ah_dms}")

        st.markdown("**Saída (cada linha: Dh_PD \\t Dh_PI \\t AH_médio DMS)**")
        st.code("\n".join(lines), language="text")

        # -------------------- Tabela de conferência -------------------------
        display_df = results.copy()
        display_df['Dh_PD_m'] = display_df['Dh_PD_m'].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )
        display_df['Dh_PI_m'] = display_df['Dh_PI_m'].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )
        display_df['AH_med_DMS'] = display_df['AH_med_DMS'].fillna("")

        st.markdown(
            """
            <div class="section-title">
                <span class="dot"></span>
                <span>Tabela de conferência (valores numéricos detalhados)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            display_df[
                [
                    'EST', 'PV',
                    'DistanciaInclinada_PD', 'DistanciaInclinada_PI',
                    'AnguloZenital_PD', 'AnguloZenital_PI',
                    'Dh_PD_m', 'Dh_PI_m', 'AH_med_DMS'
                ]
            ],
            use_container_width=True
        )

        # -------------------- Download da saída CSV -------------------------
        out_df = results[['Dh_PD_m', 'Dh_PI_m', 'AH_med_DMS']].copy()
        out_df.rename(
            columns={
                'Dh_PD_m': 'Dh_PD_m (m)',
                'Dh_PI_m': 'Dh_PI_m (m)',
                'AH_med_DMS': 'AH_med_DMS'
            },
            inplace=True
        )
        out_csv = out_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar saída (CSV)",
            data=out_csv,
            file_name="saida_topografia.csv",
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
        Versão do app: <code>1.3</code> — cabeçalho UFPE + campos de identificação do trabalho.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # fim main-card
