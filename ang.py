import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Cálculos de Estação Total", layout="wide")

st.title("📐 Cálculos de Estação Total – Tabela Dinâmica")

st.markdown(
    "Preencha a tabela abaixo. Você pode escolher quantas linhas quiser. "
    "Os cálculos serão gerados automaticamente."
)

# =========================================================
# 1) DEFINIÇÃO DO NÚMERO DE LINHAS
# =========================================================
num_linhas = st.number_input("Quantidade de linhas:", min_value=1, max_value=200, value=5)

# Estrutura padrão da tabela
colunas = [
    "EST", "PV",
    "AH_PD", "AH_PI",
    "AZ_PD", "AZ_PI",
    "DI_PD", "DI_PI"
]

# Criar dataframe
df = pd.DataFrame([[""] * len(colunas) for _ in range(num_linhas)], columns=colunas)

# Mostrar tabela editável
df_editada = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
)

st.divider()

# =========================================================
# 2) FUNÇÕES DE CÁLCULO
# =========================================================

def media_angulo(pd, pi):
    try:
        return (float(pd) + float(pi)) / 2
    except:
        return np.nan

def distancia_media(pd, pi):
    try:
        return (float(pd) + float(pi)) / 2
    except:
        return np.nan

# =========================================================
# 3) APLICAR CÁLCULOS À TABELA
# =========================================================

resultado = pd.DataFrame()
resultado["EST"] = df_editada["EST"]
resultado["PV"] = df_editada["PV"]

# Médias dos ângulos
resultado["Ángulo H Médio"] = [
    media_angulo(a, b) for a, b in zip(df_editada["AH_PD"], df_editada["AH_PI"])
]
resultado["Ángulo Z Médio"] = [
    media_angulo(a, b) for a, b in zip(df_editada["AZ_PD"], df_editada["AZ_PI"])
]

# Média das distâncias
resultado["Distância Média"] = [
    distancia_media(a, b) for a, b in zip(df_editada["DI_PD"], df_editada["DI_PI"])
]

st.subheader("📊 Resultado dos Cálculos")
st.dataframe(resultado, use_container_width=True)

# =========================================================
# 4) DOWNLOAD DO EXCEL
# =========================================================

def to_excel(df):
    from io import BytesIO
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Resultados")
    writer.close()
    return output.getvalue()

st.download_button(
    label="📥 Baixar resultados em Excel",
    data=to_excel(resultado),
    file_name="resultado_estacao_total.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
