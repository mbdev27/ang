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
# CSS - layout, cores e contraste
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Fundo geral independente do tema claro/escuro do navegador */
    body, .stApp {
        background-color: #F8F8F8 !important;
        color: #111111 !important;
    }

    .main {
        padding-top: 1rem;
    }

    /* Cabeçalho UFPE */
    .cabecalho-ufpe {
        border: 1px solid #B00020;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        background-color: #FFFFFF;
        color: #111111;
    }

    .cabecalho-ufpe h1,
    .cabecalho-ufpe h2,
    .cabecalho-ufpe h3 {
        margin: 0;
        padding: 0;
        color: #8B0014;  /* Bordô UFPE */
    }

    .cabecalho-ufpe small {
        color: #333333;
    }

    /* Campos de entrada */
    .stTextInput > label,
    .stNumberInput > label,
    .stDateInput > label,
    .stFileUploader > label {
        color: #111111 !important;
        font-weight: 500;
    }

    .stTextInput input,
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #CCCCCC;
    }

    /* Botões */
    .stButton button {
        background-color: #8B0014 !important;
        color: #FFFFFF !important;
        border-radius: 4px;
        border: 1px solid #5e000e !important;
    }
    .stButton button:hover {
        background-color: #5e000e !important;
    }

    /* DataFrames */
    .stDataFrame, .stDataEditor {
        background-color: #FFFFFF !important;
        border-radius: 4px;
        border: 1px solid #DDDDDD;
    }

    .stDataFrame table thead tr {
        background-color: #8B0014 !important;
        color: #FFFFFF !important;
    }

    .stDataFrame table tbody tr:nth-child(odd) {
        background-color: #FAFAFA !important;
    }

    .stDataFrame table tbody tr:nth-child(even) {
        background-color: #FFFFFF !important;
    }

    /* Mensagens */
    .stAlert {
        border-radius: 4px;
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
    Ex.: '145°47\'33"' ou '145 47 33' -> 145.7925...
    """
    if pd.isna(dms_str):
        return np.nan

    s = str(dms_str).strip()

    # Aceita formato com espaços: g m s
    if " " in s and "°" not in s and "'" not in s and '"' not in s:
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

    # Formatos com ° ' "
    # Remove símbolos não numéricos exceto separadores
    s = s.replace("º", "°")
    s = s.replace("’", "'").replace("´", "'")
    for ch in [" ", "  "]:
        s = s.replace(ch, "")

    g = m = sec = 0.0
    try:
        if "°" in s:
            parts = s.split("°")
            g = float(parts[0])
            resto = parts[1]
        else:
            return float(s)  # já decimal

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
    Retorna DataFrame com:
    EST, PV, Hz_PD (dec), Hz_PI (dec), Hz_medio (dec),
    DH_PD, DH_PI, DH_medio, DN_PD, DN_PI, DN_medio
    """
    df_calc = df.copy()

    # Converte ângulos para decimal
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        df_calc[col + "_dec"] = df_calc[col].apply(dms_to_decimal)

    # Convertendo distâncias para float
    for col in ["DI_PD", "DI_PI"]:
        df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce")

    # Média de direção (em graus decimais)
    df_calc["Hz_medio"] = (df_calc["Hz_PD_dec"] + df_calc["Hz_PI_dec"]) / 2.0

    # Distância horizontal e diferença de nível
    # Z em graus decimais -> radianos
    for lado in ["PD", "PI"]:
        z_dec = df_calc[f"Z_{lado}_dec"] * np.pi / 180.0
        di = df_calc[f"DI_{lado}"]

        df_calc[f"DH_{lado}"] = di * np.sin(z_dec)   # se Z é zenital
        df_calc[f"DN_{lado}"] = di * np.cos(z_dec)

    # Médias
    df_calc["DH_medio"] = (df_calc["DH_PD"] + df_calc["DH_PI"]) / 2.0
    df_calc["DN_medio"] = (df_calc["DN_PD"] + df_calc["DN_PI"]) / 2.0

    # Arredonda para saída (distâncias em m com 3 casas)
    for col in ["DH_PD", "DH_PI", "DH_medio", "DN_PD", "DN_PI", "DN_medio"]:
        df_calc[col] = df_calc[col].round(3)

    # Monta resultado para exibição
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
        # Se tiver um arquivo de imagem do brasão, coloque o caminho aqui
        # st.image("brasao_ufpe.png", use_column_width=True)
        st.write("")

    with col_titulo:
        st.markdown(
            """
            <div class="cabecalho-ufpe">
                <h2>Universidade Federal de Pernambuco - UFPE</h2>
                <h3>Centro de Tecnologia e Geociências (CTG)</h3>
                <h3>Curso de Engenharia Cartográfica e de Agrimensura</h3>
                <small>Aplicação para Cálculo de Direções Horizontais e Distâncias</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")  # espaçamento

    # ----------------- Campos de identificação ---------------
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

        Os ângulos podem estar em formato `g°m′s″` (por ex.: `145°47'33"`)
        ou `g m s` (ex.: `145 47 33`).  
        As distâncias devem estar em metros, com ponto decimal (ex.: `25.365`).
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

        st.subheader("2. Tabela de conferência dos dados importados")
        st.dataframe(df, use_container_width=True)

        # ------------- Validação de colunas -------------------
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
                "Z_PD, Z_PI, DI_PD, DI_PI\n\n"
                f"Faltando: {', '.join(faltando)}"
            )
            return

        # Checar dados vazios nas colunas críticas
        if df[colunas_obrigatorias].isna().any().any():
            st.warning(
                "Foram encontrados valores vazios em algumas das colunas "
                "obrigatórias. Essas linhas poderão gerar resultados vazios "
                "nos cálculos."
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

            # Opção de download dos resultados
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
