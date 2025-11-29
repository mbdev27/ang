"""
Streamlit app: Cálculos de Ângulos e Distâncias (Estação Total)
Arquivo: streamlit_calc_angulo_distancia.py
Autor: Gerado com ChatGPT
Descrição: Aplicativo leve para calcular Distância Horizontal (Dh) e componente vertical (Δh)
a partir da Distância Inclinada (DI) e do Ângulo Zenital (Z). Suporta entrada manual, colagem/CSV e mostra fórmulas em LaTeX.
Como usar: rodar `streamlit run streamlit_calc_angulo_distancia.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from math import sin, cos, radians

st.set_page_config(page_title="Cálculos Angulares — Estação Total", layout="wide")

st.title("Cálculos de Ângulos e Distâncias — Estação Total")
st.markdown(
    """
    Ferramenta simples para calcular **Distância Horizontal** e **Componente Vertical** a partir da **Distância Inclinada (DI)** e do **Ângulo Zenital (Z)**.

    Fórmulas principais (Z em graus):

    - Distância Horizontal: $D_h = DI \cdot \sin(Z)$
    - Componente vertical (diferença altura): $\Delta h = DI \cdot \cos(Z)$

    Insira sua planilha (CSV/Excel) ou cole dados no campo de edição. Aceitamos ângulos no formato DMS (ex: `89°48\'20\"`, `89 48 20`, `89:48:20`) ou em graus decimais (`89.8056`).
    """
)

st.header("1) Carregar dados")

uploaded = st.file_uploader("Envie um arquivo CSV ou Excel (opcional)", type=["csv", "xls", "xlsx"]) 

# sample dataframe template
sample = pd.DataFrame({
    'EST': ['P1','P1'],
    'PV': ['P2','P3'],
    'DI_m': [25.365, 26.285],
    'Z': ["89°48'20\"","89°36'31\""]
})

# download model template
st.download_button(
    label="Baixar modelo de tabela (CSV)",
    data=sample.to_csv(index=False).encode('utf-8'),
    file_name='modelo_tabela_topografia.csv',
    mime='text/csv'
)

sample = pd.DataFrame({
    'EST': ['P1','P1'],
    'PV': ['P2','P3'],
    'DI_m': [25.365, 26.285],
    'Z': ["89°48'20\"","89°36'31\""]
})

st.markdown("**Modelo de colunas esperado:** `DI_m` (Distância inclinada, metros) e `Z` (ângulo zenital em DMS ou decimal).")
if st.checkbox("Mostrar exemplo de tabela"):
    st.dataframe(sample)

# Read data
df = None
if uploaded is not None:
    try:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

st.markdown("Ou cole/cole os dados (CSV) abaixo)")
text_input = st.text_area("Colar CSV aqui (opcional)", height=120)
if text_input.strip() != "" and df is None:
    try:
        df = pd.read_csv(io.StringIO(text_input))
    except Exception as e:
        st.error(f"Erro ao interpretar o CSV colado: {e}")

# If no data provided, create an empty df for manual entry
if df is None:
    st.info("Nenhuma tabela carregada — você pode entrar com valores individuais abaixo ou clicar em 'Usar tabela de exemplo'.")
    if st.button("Usar tabela de exemplo"):
        df = sample.copy()

st.header("2) Conversão de ângulos e cálculos")

# helper: parse DMS to decimal degrees
angle_pattern = re.compile(r"(-?\d+)[°:\s]+(\d+)[\':\s]+(\d+(?:\.\d+)?)\"?\s*$")
angle_pattern2 = re.compile(r"^-?\d+(?:\.\d+)?$")


def dms_to_decimal(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    # try common separators
    s = s.replace('º', '°')
    s = s.replace('’', "'")
    s = s.replace('″', '"')
    s = s.replace(':', ' ')
    s = s.replace('  ', ' ')
    # match D M S
    m = re.search(r"(-?\d+)\D+(\d+)\D+(\d+(?:[\.,]\d+)?)", s)
    if m:
        deg = float(m.group(1))
        minu = float(m.group(2))
        sec = float(m.group(3).replace(',', '.'))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu / 60.0 + sec / 3600.0)
    # match single decimal number
    m2 = re.search(r"^-?\d+(?:[\.,]\d+)?$", s)
    if m2:
        return float(s.replace(',', '.'))
    # fallback: try to extract numbers
    nums = re.findall(r"-?\d+(?:[\.,]\d+)?", s)
    if len(nums) == 3:
        deg, minu, sec = nums
        deg = float(deg); minu = float(minu); sec = float(sec)
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu / 60.0 + sec / 3600.0)
    # can't parse
    raise ValueError(f"Formato de ângulo desconhecido: '{s}'")


if df is not None:
    df = df.copy()
    # ensure DI column exists; allow multiple possible names
    possible_di = [c for c in df.columns if c.lower().strip() in ['di','di_m','distancia','distancia_inclinada','distancia inclinada','distancia_inclinada(di)']]
    possible_z = [c for c in df.columns if c.lower().strip() in ['z','zenital','angulo_zenital','ângulo_zenital','observacoes','v']]

    if len(possible_di) == 0:
        st.warning('Coluna DI não encontrada; insira DI manualmente abaixo ou renomeie sua coluna para "DI_m"')
    else:
        di_col = possible_di[0]
        df.rename(columns={di_col: 'DI_m'}, inplace=True)

    if len(possible_z) == 0:
        st.warning('Coluna Z (zenital) não encontrada; você pode inserir manualmente ou incluir uma coluna "Z" na sua tabela.')
    else:
        z_col = possible_z[0]
        df.rename(columns={z_col: 'Z'}, inplace=True)

    # allow manual single-row input
    with st.expander('Entrada manual (uma linha)'):
        col1, col2 = st.columns(2)
        with col1:
            di_manual = st.number_input('DI (m)', value=25.365, step=0.001, format="%.3f")
        with col2:
            z_manual = st.text_input('Z (graus ou DMS)', value="89°48'20\"")
        if st.button('Adicionar linha manual'):
            new = {'EST': 'manual', 'PV': '', 'DI_m': di_manual, 'Z': z_manual}
            if df is None:
                df = pd.DataFrame([new])
            else:
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

    # parse DI and Z
    # ensure DI numeric
    try:
        df['DI_m'] = pd.to_numeric(df['DI_m'], errors='coerce')
    except Exception:
        pass

    # parse Z to decimal degrees
    z_dec = []
    for i, val in enumerate(df.get('Z', pd.Series([np.nan]*len(df)))):
        try:
            zdeg = dms_to_decimal(val)
        except Exception as e:
            zdeg = np.nan
        z_dec.append(zdeg)
    df['Z_deg'] = z_dec

    # compute Dh and delta_h
    def compute_row(di, zdeg):
        if pd.isna(di) or pd.isna(zdeg):
            return (np.nan, np.nan)
        zrad = radians(zdeg)
        dh = di * sin(zrad)
        dh_v = di * cos(zrad)  # componente vertical (Delta h)
        return (dh, dh_v)

    res = df.apply(lambda r: compute_row(r.get('DI_m', np.nan), r.get('Z_deg', np.nan)), axis=1)
    df[['Dh_m', 'Delta_h_m']] = pd.DataFrame(res.tolist(), index=df.index)

    # formatting
    df['DI_m'] = df['DI_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    df['Z_deg'] = df['Z_deg'].map(lambda x: f"{x:.6f}" if pd.notna(x) else '')
    df['Dh_m'] = df['Dh_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    df['Delta_h_m'] = df['Delta_h_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')

    st.subheader('Tabela de resultados')
    st.dataframe(df)

    # download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Baixar CSV com resultados', data=csv, file_name='resultados_angulos_distancias.csv', mime='text/csv')

    st.subheader('Fórmulas (exibidas)')
    st.latex(r"D_h = DI \cdot \sin(Z)")
    st.latex(r"\Delta h = DI \cdot \cos(Z)")

    st.markdown("---")

    st.header('Visualização rápida')
    try:
        # plot DI vs Dh simples
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        numeric_DIs = pd.to_numeric(df['DI_m'], errors='coerce')
        numeric_Dhs = pd.to_numeric(df['Dh_m'], errors='coerce')
        ax.scatter(numeric_DIs, numeric_Dhs)
        ax.set_xlabel('DI (m)')
        ax.set_ylabel('Dh (m)')
        ax.set_title('DI vs Dh')
        st.pyplot(fig)
    except Exception:
        pass

st.header('3) Observações e deploy')
st.markdown(
    """
    - O aplicativo tenta reconhecer colunas com nomes comuns (ex.: `DI_m`, `DI`, `distancia_inclinada`, `Z`, `Zenital`, `Observacoes`).
    - Aceita ângulos no formato DMS (ex: `89°48'20"`) ou em decimal (ex: `89.8056`).
    - Dependências leves: `streamlit`, `pandas`, `numpy`, `matplotlib` (opcional para o gráfico).

    **Para publicar no GitHub e rodar no Streamlit Cloud**:

    1. Crie um repositório no GitHub e adicione este arquivo como `streamlit_app.py` (ou `streamlit_calc_angulo_distancia.py`).
    2. Crie `requirements.txt` com: `streamlit\npandas\nnumpy\nmatplotlib` (uma dependência por linha).
    3. Faça commit e push.
    4. No Streamlit Cloud, conecte o repositório e escolha o arquivo principal. O deploy deve rodar automaticamente.

    Se quiser, eu gero também o `requirements.txt` e um `README.md` prontos para você colocar no repositório.
    """
)

st.info('App pronto — ajuste nomes de colunas no seu CSV para "DI_m" e "Z" se algo não for detectado automaticamente.')
