import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Configuração básica da página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Cálculo de Direções e Distâncias - UFPE",
    layout="wide"
)

# ------------------------------------------------------------
# CSS - estilo mais orgânico / suave
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Fundo geral bem claro, mais "orgânico" */
    body, .stApp {
        background: linear-gradient(135deg, #f7f7f7 0%, #fdfdfd 40%, #f5f5f5 100%) !important;
        color: #1a1a1a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .main {
        padding-top: 0.5rem;
    }

    /* Cabeçalho UFPE com leve sombra e cantos arredondados */
    .cabecalho-ufpe {
        border-radius: 12px;
        padding: 1rem 1.4rem;
        background: #ffffffdd;
        backdrop-filter: blur(3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }

    .cabecalho-ufpe h2 {
        margin: 0 0 0.25rem 0;
        padding: 0;
        color: #7d1220; /* bordô UFPE mais suave */
        font-size: 1.4rem;
        letter-spacing: 0.02em;
    }

    .cabecalho-ufpe h3 {
        margin: 0;
        padding: 0;
        color: #a32a36;
        font-size: 1rem;
        font-weight: 600;
    }

    .cabecalho-ufpe small {
        display: block;
        margin-top: 0.4rem;
        color: #444444;
        font-size: 0.85rem;
    }

    /* Expander mais suave */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #7d1220 !important;
    }

    /* Labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stDateInput > label,
    .stFileUploader > label {
        color: #2d2d2d !important;
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* Inputs com bordas arredondadas */
    .stTextInput input,
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border-radius: 999px;
        border: 1px solid #d4d4d4;
        padding: 0.35rem 0.8rem;
        font-size: 0.9rem;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #a32a36 !important;
        box-shadow: 0 0 0 1px rgba(163,42,54,0.25);
    }

    /* Botões com estilo "pill" */
    .stButton button {
        background: linear-gradient(135deg, #a32a36, #7d1220) !important;
        color: #ffffff !important;
        border-radius: 999px;
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

    /* DataFrames em "cartão" */
    .stDataFrame, .stDataEditor {
        background-color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 4px 16px rgba(0,0,0,0.03);
        padding: 0.2rem 0.4rem;
    }

    .stDataFrame table thead tr {
        background-color: #f5e6e8 !important;
        color: #5b101d !important;
        font-weight: 600;
    }

    .stDataFrame table tbody tr:nth-child(odd) {
        background-color: #fafafa !important;
    }
    .stDataFrame table tbody tr:nth-child(even) {
        background-color: #ffffff !important;
    }

    .stDataFrame table tbody tr:hover {
        background-color: #f3f0f0 !important;
    }

    /* Alertas mais arredondados */
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def dms_to_decimal(dms_str: str) -> float:
    """
    Converte um ângulo em string (g°m′s″ ou 'g m s') para graus decimais.
    Ex.: '145°47\\'33\"' ou '145 47 33' -> 145.7925...
    Retorna NaN se não conseguir converter.
    """
    if pd.isna(dms_str):
        return np.nan

    s = str(dms_str).strip()
    if s == "":
        return np.nan

    # Formato "g m s"
    if " " in s and all(ch not in s for ch in ["°", "º", "'", "’", '"']):
        try:
            g, m, sec = s.split()
            g = float(g)
            m = float(m)
            sec = float(sec)
            sinal = -1 if g < 0 else 1
            g = abs(g)
            return sinal * (g + m/60 + sec/3600)
        except Exception:
            return np.nan

    # Normaliza símbolos
    s = s.replace("º", "°")
    s = s.replace("’", "'").replace("´", "'")
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.replace(" ", "")

    g = m = sec = 0.0
    try:
        if "°" in s:
            parts = s.split("°")
            g = float(parts[0])
            resto = parts[1]
        else:
            return float(s)

        if "'" in resto:
            parts = resto.split("'")
            m = float(parts[0])
            resto2 = parts[1]
        else:
            return g + m/60.0

        if '"' in resto2:
            parts = resto2.split('"')
            sec = float(parts[0])
        else:
            sec = float(resto2)

        sinal = -1 if g < 0 else 1
        g = abs(g)
        return sinal * (g + m/60.0 + sec/3600.0)
    except Exception:
        return np.nan


def calcula_resultados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame com colunas obrigatórias:
    EST, PV, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI
    """
    df_calc = df.copy()

    # Converte ângulos para decimal
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        df_calc[col + "_dec"] = df_calc[col].apply(dms_to_decimal)

    # Converte distâncias
    for col in ["DI_PD", "DI_PI"]:
        df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce")

    # Hz médio (graus)
    df_calc["Hz_medio"] = (df_calc["Hz_PD_dec"] + df_calc["Hz_PI_dec"]) / 2.0

    # Distâncias horizontal e diferença de nível
    for lado in ["PD", "PI"]:
        z_dec = df_calc[f"Z_{lado}_dec"] * np.pi / 180.0
        di = df_calc[f"DI_{lado}"]
        df_calc[f"DH_{lado}"] = di * np.sin(z_dec)
        df_calc[f"DN_{lado}"] = di * np.cos(z_dec)

    df_calc["DH_medio"] = (df_calc["DH_PD"] + df_calc["DH_PI"]) / 2.0
    df_calc["DN_medio"] = (df_calc["DN_PD"] + df_calc["DN_PI"]) / 2.0

    # Arredonda distâncias
    for col in ["DH_PD", "DH_PI", "DH_medio", "DN_PD", "DN_PI", "DN_medio"]:
        df_calc[col] = df_calc[col].round(3)

    resultado = df_calc[[
        "EST", "PV",
        "Hz_PD_dec", "Hz_PI_dec", "Hz_medio",
        "DH_PD", "DH_PI", "DH_medio",
        "DN_PD", "DN_PI", "DN_medio"
    ]].rename(columns={
        "Hz_PD_dec": "Hz_PD (graus)",
        "Hz_PI_dec": "Hz_PI (graus)",
        "Hz_medio": "Hz_médio (graus)",
        "DH_PD": "DH_PD (m)",
        "DH_PI": "DH_PI (m)",
        "DH_medio": "DH_médio (m)",
        "DN_PD": "DN_PD (m)",
        "DN_PI": "DN_PI (m)",
        "DN_medio": "DN_médio (m)"
    })

    return resultado


# ------------------------------------------------------------
# Aplicação principal
# ------------------------------------------------------------
def main():
    # ----------------- Cabeçalho UFPE ------------------------
    col_logo, col_titulo = st.columns([1, 4])

    with col_logo:
        # Se tiver o brasão em imagem, descomente:
        # st.image("brasao_ufpe.png", use_column_width=True)
        st.write("")

    with col_titulo:
        st.markdown(
            """
            <div class="cabecalho-ufpe">
                <h2>Universidade Federal de Pernambuco - UFPE</h2>
                <h3>Centro de Tecnologia e Geociências · Engenharia Cartográfica e de Agrimensura</h3>
                <small>Aplicação para Cálculo de Direções Horizontais e Distâncias com Estação Total</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ----------------- Informações da campanha ---------------
    with st.expander("Informações da campanha", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            professor = st.text_input("Professor")
        with c2:
            local = st.text_input("Local")
        with c3:
            equipamento = st.text_input("Equipamento")
        with c4:
            data_campanha = st.date_input("Data")
        with c5:
            patrimonio = st.text_input("Patrimônio")

    st.write("")

    # ----------------- Upload de planilha --------------------
    st.subheader("1. Importar dados de campo")

    st.markdown(
        """
        **Modelo esperado de planilha (.xlsx)** – cabeçalhos (linha 1):

        - `EST`
        - `PV`
        - `Hz_PD`  – Ângulo Horizontal PD  
        - `Hz_PI`  – Ângulo Horizontal PI  
        - `Z_PD`   – Ângulo Zenital PD  
        - `Z_PI`   – Ângulo Zenital PI  
        - `DI_PD`  – Distância Inclinada PD (m)  
        - `DI_PI`  – Distância Inclinada PI (m)  

        Ângulos em formato `g°m′s″` (ex.: `145°47'33"`) ou `g m s` (ex.: `145 47 33`).  
        Distâncias em metros, com ponto decimal (ex.: `25.365`).
        """,
        unsafe_allow_html=False,
    )

    uploaded_file = st.file_uploader(
        "Selecione a planilha de dados (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            return

        st.subheader("2. Conferência dos dados importados")
        st.dataframe(df, use_container_width=True)

        colunas_obrigatorias = [
            "EST", "PV",
            "Hz_PD", "Hz_PI",
            "Z_PD", "Z_PI",
            "DI_PD", "DI_PI"
        ]
        faltando = [c for c in colunas_obrigatorias if c not in df.columns]

        if faltando:
            st.error(
                "A planilha não está no formato esperado.\n\n"
                "Cabeçalhos obrigatórios: EST, PV, Hz_PD, Hz_PI, "
                "Z_PD, Z_PI, DI_PD, DI_PI.\n\n"
                f"Faltando: {', '.join(faltando)}"
            )
            return

        if df[colunas_obrigatorias].isna().any().any():
            st.warning(
                "Existem valores vazios em colunas obrigatórias. "
                "Essas linhas podem gerar resultados vazios nos cálculos."
            )

        st.write("")
        st.subheader("3. Cálculos e resultados")

        if st.button("Executar cálculos"):
            try:
                df_result = calcula_resultados(df)
            except Exception as e:
                st.error(f"Erro durante os cálculos: {e}")
                return

            st.success("Cálculos concluídos com sucesso.")
            st.dataframe(df_result, use_container_width=True)

            csv = df_result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Baixar resultados em CSV",
                data=csv,
                file_name="resultados_direcoes_distancias.csv",
                mime="text/csv",
            )
    else:
        st.info("Faça o upload da planilha para visualizar e calcular os resultados.")


if __name__ == "__main__":
    main()
