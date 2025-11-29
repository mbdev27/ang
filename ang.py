"""
Streamlit app: Cálculos de Ângulos e Distâncias (Estação Total)
Arquivo: app.py
Autor: Gerado com ChatGPT

Descrição:
Aplicativo para entrada interativa de linhas (EST/PV) com PD/PI de ângulos e DI,
cálculo automático de DI médio, Distância Horizontal (Dh) e componente vertical (V).

Como usar:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math
from math import sin, cos, radians

# =====================================================================
# CONFIG DA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Calculadora Topográfica — Estação Total",
    layout="wide"
)

st.title("📐 Calculadora de Ângulos e Distâncias — Estação Total")

st.markdown("""
Insira os pares **PD / PI** de **Ângulo Horizontal**, **Ângulo Zenital** e **Distância Inclinada**.

Você pode:
- **Baixar um modelo Excel**, preencher e enviar;
- **Editar diretamente na tabela** abaixo, adicionando linhas livremente.

Os ângulos podem estar em:
- **DMS** (ex.: `89°48'20"`)
- **Decimal** (ex.: `89.8056`)
""")

# =====================================================================
# MODELO PARA DOWNLOAD
# =====================================================================
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

    st.markdown("**Baixar modelo de tabela (Excel):**")
    st.download_button(
        "Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_estacao_total.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except:
    # fallback CSV
    csv_bytes = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Baixar modelo CSV",
        data=csv_bytes,
        file_name="modelo_estacao_total.csv",
        mime='text/csv'
    )

st.markdown("---")

# =====================================================================
# UPLOAD DO ARQUIVO
# =====================================================================
uploaded = st.file_uploader(
    "Envie o Excel preenchido (opcional)",
    type=["xls", "xlsx", "csv"]
)

# =====================================================================
# FUNÇÃO PARA PARSE DE DMS
# =====================================================================
angle_re = re.compile(r"(-?\d+)\D+(\d+)\D+(\d+(?:[.,]\d+)?)")
num_re = re.compile(r"^-?\d+(?:[.,]\d+)?$")

def parse_angle_to_decimal(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return np.nan

    # normaliza símbolos
    s = s.replace('º', '°').replace('\u2019', "'").replace('\u201d', '"')
    s = s.replace(':', ' ').replace('\t', ' ')
    s = re.sub('\s+', ' ', s)

    # tenta DMS
    m = angle_re.search(s)
    if m:
        deg = float(m.group(1))
        minu = float(m.group(2))
        sec = float(m.group(3).replace(',', '.'))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu/60 + sec/3600)

    # decimal simples
    if num_re.match(s.replace(' ', '')):
        return float(s.replace(',', '.'))

    # tenta extrair 3 números
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s)
    if len(nums) == 3:
        deg, minu, sec = nums
        deg = float(deg); minu = float(minu); sec = float(sec.replace(',', '.'))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu/60 + sec/3600)

    return np.nan


# =====================================================================
# CARREGAMENTO DO DF
# =====================================================================
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"Arquivo '{uploaded.name}' carregado com {len(df)} linhas.")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        df = template_df.copy()
else:
    # cria tabela com 5 linhas iniciais
    df = template_df.copy()
    df = pd.concat([df, pd.DataFrame([{} for _ in range(4)])], ignore_index=True)

# =====================================================================
# NORMALIZA NAMES
# =====================================================================
required = [
    'EST','PV',
    'AnguloHorizontal_PD','AnguloHorizontal_PI',
    'AnguloZenital_PD','AnguloZenital_PI',
    'DistanciaInclinada_PD','DistanciaInclinada_PI'
]

col_map = {}
for c in df.columns:
    low = c.lower()

    if 'horizontal' in low:
        if 'pd' in low: col_map[c] = 'AnguloHorizontal_PD'
        elif 'pi' in low: col_map[c] = 'AnguloHorizontal_PI'

    elif 'zenit' in low:
        if 'pd' in low: col_map[c] = 'AnguloZenital_PD'
        elif 'pi' in low: col_map[c] = 'AnguloZenital_PI'

    elif 'dist' in low and ('di' in low or 'inclina' in low):
        if 'pd' in low: col_map[c] = 'DistanciaInclinada_PD'
        elif 'pi' in low: col_map[c] = 'DistanciaInclinada_PI'

if col_map:
    df = df.rename(columns=col_map)

for col in required:
    if col not in df.columns:
        df[col] = ''

# =====================================================================
# EDITOR INTERATIVO
# =====================================================================
st.markdown("### Tabela (edite; clique em + para adicionar linhas):")

edited = st.data_editor(
    df[required],
    num_rows="dynamic",
    key="editor_topografia"
)

# =====================================================================
# CÁLCULOS
# =====================================================================
results = edited.copy()

# ângulos
for c in ['AnguloHorizontal_PD','AnguloHorizontal_PI','AnguloZenital_PD','AnguloZenital_PI']:
    results[c + '_deg'] = results[c].apply(parse_angle_to_decimal)

# DI
for c in ['DistanciaInclinada_PD','DistanciaInclinada_PI']:
    results[c + '_m'] = pd.to_numeric(results[c], errors='coerce')

# Médias
results['AH_med_deg'] = results[['AnguloHorizontal_PD_deg','AnguloHorizontal_PI_deg']].mean(axis=1)
results['AZ_med_deg'] = results[['AnguloZenital_PD_deg','AnguloZenital_PI_deg']].mean(axis=1)
results['DI_med_m'] = results[['DistanciaInclinada_PD_m','DistanciaInclinada_PI_m']].mean(axis=1)

# Distância Horizontal e Vertical
results['AZ_med_rad'] = results['AZ_med_deg'].apply(lambda x: math.radians(x) if pd.notna(x) else np.nan)
results['Dh_m'] = results.apply(
    lambda r: r['DI_med_m'] * math.sin(r['AZ_med_rad'])
    if pd.notna(r['DI_med_m']) and pd.notna(r['AZ_med_rad']) else np.nan,
    axis=1
)
results['V_m'] = results.apply(
    lambda r: r['DI_med_m'] * math.cos(r['AZ_med_rad'])
    if pd.notna(r['DI_med_m']) and pd.notna(r['AZ_med_rad']) else np.nan,
    axis=1
)

# =====================================================================
# EXIBIÇÃO
# =====================================================================
display_cols = [
    'EST','PV',
    'AnguloHorizontal_PD','AnguloHorizontal_PI',
    'AnguloZenital_PD','AnguloZenital_PI',
    'DistanciaInclinada_PD','DistanciaInclinada_PI',
    'DI_med_m','Dh_m','V_m'
]

results_display = results.copy()
fmt = lambda x: f"{x:.3f}" if pd.notna(x) else ""

results_display['DI_med_m'] = results_display['DI_med_m'].map(fmt)
results_display['Dh_m'] = results_display['Dh_m'].map(fmt)
results_display['V_m'] = results_display['V_m'].map(fmt)

st.markdown("---")
st.subheader("Resultados calculados")
st.dataframe(results_display[display_cols], use_container_width=True)

# =====================================================================
# DOWNLOAD
# =====================================================================
out_csv = results.to_csv(index=False).encode('utf-8')
st.download_button(
    'Baixar resultados (CSV)',
    data=out_csv,
    file_name='resultados_topografia.csv',
    mime='text/csv'
)

st.markdown("---")

st.header("Notas rápidas")
st.markdown("""
- Se o Excel subir com nomes estranhos, o app tenta corrigir automaticamente.
- DI, Dh e V são calculados apenas quando todos os valores estão preenchidos.
- Lembre-se de instalar `openpyxl` para permitir exportação em Excel.
""")
