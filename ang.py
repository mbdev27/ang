"""
Streamlit app: Cálculos de Ângulos e Distâncias (Estação Total)
Arquivo: app.py
Autor: Gerado com ChatGPT
Descrição: Aplicativo para entrada interativa de linhas (EST/PV) com PD/PI de ângulos e DI, cálculo automático de DI médio, Distância Horizontal (Dh) e componente vertical (V).
Como usar: rodar `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math
from math import sin, cos, radians

st.set_page_config(page_title="Calculadora Topográfica — Estação Total", layout="wide")

st.title("📐 Calculadora de Ângulos e Distâncias — Estação Total")
st.markdown(
    """
    Insira os pares PD / PI de Ângulo Horizontal, Ângulo Zenital e Distância Inclinada.

    Você pode:
    - Baixar um modelo Excel, preencher e subir; ou
    - Editar as linhas diretamente nesta tabela (adicione quantas linhas quiser).

    Os ângulos aceitos podem estar em **DMS** (ex: `89°48'20"`), com separadores `° ' "` ou `:` ou espaços, ou em decimal (`89.8056`).
    """
)

# ---------------------- template and download --------------------------------
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
    st.markdown("**Modelo de tabela Excel para download:**")
    st.download_button(
        "Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_estacao_total.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception:
    # fallback: offer CSV
    csv_bytes = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar modelo CSV", data=csv_bytes, file_name="modelo_estacao_total.csv", mime='text/csv')

st.markdown("---")

# ---------------------- file upload -------------------------------------------
uploaded = st.file_uploader("Envie o Excel preenchido (opcional)", type=["xls", "xlsx", "csv"])

# ---------------------- helper: parse DMS ------------------------------------
angle_re = re.compile(r"(-?\\d+)[^\\d\\-]+(\\d+)[^\\d\\-]+(\\d+(?:[\\.,]\\d+)?)")
num_re = re.compile(r"^-?\\d+(?:[\\.,]\\d+)?$")


def parse_angle_to_decimal(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return np.nan
    s = s.replace('º', '°')
    s = s.replace('\u2019', "'")
    s = s.replace('\u201d', '"')
    s = s.replace(':', ' ')
    s = s.replace('\t', ' ')
    s = re.sub('\s+', ' ', s)
    # try D M S groups
    m = re.search(r"(-?\\d+)\\D+(\\d+)\\D+(\\d+(?:[\\.,]\\d+)?)", s)
    if m:
        deg = float(m.group(1))
        minu = float(m.group(2))
        sec = float(m.group(3).replace(',', '.'))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu / 60.0 + sec / 3600.0)
    # try simple decimal
    m2 = re.search(r"-?\\d+(?:[\\.,]\\d+)?", s)
    if m2 and num_re.match(s.replace(' ', '')):
        return float(s.replace(',', '.'))
    # else try to extract three numbers
    nums = re.findall(r"-?\\d+(?:[\\.,]\\d+)?", s)
    if len(nums) == 3:
        deg, minu, sec = nums
        deg = float(deg); minu = float(minu); sec = float(sec)
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu / 60.0 + sec / 3600.0)
    # can't parse
    return np.nan

# ---------------------- load dataframe ---------------------------------------
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
    df = template_df.copy()
    # give user an empty few rows to start
    df = pd.concat([df]*5, ignore_index=True)

# Normalize expected column names (tolerant)
col_map = {}
for c in df.columns:
    low = c.strip().lower()
    if 'anguloh' in low or 'horizontal' in low:
        if 'pd' in low or low.endswith('_pd'):
            col_map[c] = 'AnguloHorizontal_PD'
        elif 'pi' in low or low.endswith('_pi'):
            col_map[c] = 'AnguloHorizontal_PI'
        else:
            # if ambiguous, try keep as-is
            col_map[c] = c
    if 'zenit' in low or 'zenital' in low:
        if 'pd' in low or low.endswith('_pd'):
            col_map[c] = 'AnguloZenital_PD'
        elif 'pi' in low or low.endswith('_pi'):
            col_map[c] = 'AnguloZenital_PI'
        else:
            col_map[c] = c
    if 'dist' in low and 'di' in low or 'distanciainclinada' in low.replace(' ', '') or 'distancia_inclinada' in low:
        if 'pd' in low or low.endswith('_pd'):
            col_map[c] = 'DistanciaInclinada_PD'
        elif 'pi' in low or low.endswith('_pi'):
            col_map[c] = 'DistanciaInclinada_PI'
        else:
            # try detect PD/PI in header
            if '_pd' in low:
                col_map[c] = 'DistanciaInclinada_PD'
            elif '_pi' in low:
                col_map[c] = 'DistanciaInclinada_PI'
            else:
                col_map[c] = c

# apply renames if any
if len(col_map) > 0:
    try:
        df = df.rename(columns=col_map)
    except Exception:
        pass

# Ensure all required columns exist
required = ['EST','PV','AnguloHorizontal_PD','AnguloHorizontal_PI','AnguloZenital_PD','AnguloZenital_PI','DistanciaInclinada_PD','DistanciaInclinada_PI']
for col in required:
    if col not in df.columns:
        df[col] = ''

st.markdown("### Tabela (edite as células; clique em + para adicionar linhas)")
# Use data editor if available
try:
    edited = st.data_editor(df[required], num_rows="dynamic")
except Exception:
    # fallback
    edited = st.experimental_data_editor(df[required])

# After editing, parse angles and distances
results = edited.copy()
# parse angles
for c in ['AnguloHorizontal_PD','AnguloHorizontal_PI','AnguloZenital_PD','AnguloZenital_PI']:
    results[c + '_deg'] = results[c].apply(parse_angle_to_decimal)
# parse distances
for c in ['DistanciaInclinada_PD','DistanciaInclinada_PI']:
    results[c + '_m'] = pd.to_numeric(results[c], errors='coerce')

# compute means and derived
results['AH_med_deg'] = results[['AnguloHorizontal_PD_deg','AnguloHorizontal_PI_deg']].mean(axis=1, skipna=True)
results['AZ_med_deg'] = results[['AnguloZenital_PD_deg','AnguloZenital_PI_deg']].mean(axis=1, skipna=True)
results['DI_med_m'] = results[['DistanciaInclinada_PD_m','DistanciaInclinada_PI_m']].mean(axis=1, skipna=True)

# compute Dh and V
results['AZ_med_rad'] = results['AZ_med_deg'].apply(lambda x: math.radians(x) if pd.notna(x) else np.nan)
results['Dh_m'] = results.apply(lambda r: (r['DI_med_m'] * math.sin(r['AZ_med_rad'])) if pd.notna(r['DI_med_m']) and pd.notna(r['AZ_med_rad']) else np.nan, axis=1)
results['V_m'] = results.apply(lambda r: (r['DI_med_m'] * math.cos(r['AZ_med_rad'])) if pd.notna(r['DI_med_m']) and pd.notna(r['AZ_med_rad']) else np.nan, axis=1)

# format for display
display_cols = ['EST','PV','AnguloHorizontal_PD','AnguloHorizontal_PI','AnguloZenital_PD','AnguloZenital_PI','DistanciaInclinada_PD','DistanciaInclinada_PI','DI_med_m','Dh_m','V_m']
results_display = results.copy()
results_display['DI_med_m'] = results_display['DI_med_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')
results_display['Dh_m'] = results_display['Dh_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')
results_display['V_m'] = results_display['V_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')

st.markdown("---")
st.subheader("Resultados calculados")
st.dataframe(results_display[display_cols], use_container_width=True)

# allow download
out_csv = results.to_csv(index=False).encode('utf-8')
st.download_button('Baixar resultados (CSV)', data=out_csv, file_name='resultados_topografia.csv', mime='text/csv')

st.markdown("---")
st.header("Notas rápidas")
st.markdown(
    """
    - Se o Excel preenchido não carregou corretamente, verifique os nomes das colunas.
    - O app tenta mapear variações nos nomes (por exemplo: `AnguloHorizontal_PD`, `Angulo Horizontal PD`, etc.).
    - Dependência para gerar o modelo `.xlsx`: `openpyxl` (adicione ao requirements.txt).
    """
)

# End
