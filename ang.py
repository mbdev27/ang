# app_complete.py
# Streamlit app completo: Cálculos de Ângulos e Distâncias (Estação Total)
# Coloque este arquivo como app.py e rode com: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math

st.set_page_config(page_title="Calculadora Topográfica — Estação Total", layout="wide")
st.title("📐 Calculadora de Ângulos e Distâncias — Estação Total")

st.markdown(
    """
    Insira os pares **PD / PI** de **Ângulo Horizontal**, **Ângulo Zenital** e **Distância Inclinada**.
    - Baixe o modelo Excel, preencha e faça upload;  
    - ou edite diretamente a tabela (adicione/remova linhas).
    A saída final exibirá: **Dh_PD (m)** \t **Dh_PI (m)** \t **Ângulo Horizontal médio (DMS)**.
    """
)

# -------------------- template para download (Excel) -------------------------
template_df = pd.DataFrame({
    'EST': ['P1'],
    'PV': ['P2'],
    'AnguloHorizontal_PD': [''],  # AH_PD - em DMS ou decimal
    'AnguloHorizontal_PI': [''],  # AH_PI
    'AnguloZenital_PD': [''],     # AZ_PD - zenital (DMS ou decimal)
    'AnguloZenital_PI': [''],     # AZ_PI
    'DistanciaInclinada_PD': [''],# DI_PD (m)
    'DistanciaInclinada_PI': [''] # DI_PI (m)
})

excel_bytes = io.BytesIO()
try:
    template_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_estacao_total.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception:
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar modelo CSV", data=csv_bytes, file_name="modelo_estacao_total.csv", mime="text/csv")

st.markdown("---")

# -------------------- upload de arquivo -------------------------------------
uploaded = st.file_uploader("Envie a planilha preenchida (opcional)", type=["xlsx", "xls", "csv"])

# -------------------- parsing de ângulos (DMS/decimal) -----------------------
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
    s = s.replace("º", "°").replace("\u2019", "'").replace("\u201d", '"')
    s = s.replace(":", " ").replace("\t", " ")
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
        deg = float(deg); minu = float(minu); sec = float(str(sec).replace(",", "."))
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minu/60.0 + sec/3600.0)
    return np.nan

def decimal_to_dms(angle):
    """Converte grau decimal para string DMS formatada: G°MM'SS\".
       G pode ser negativo. Segundos arredondados; ajuste de overflow."""
    if pd.isna(angle):
        return ""
    sign = "-" if angle < 0 else ""
    a = abs(float(angle))
    g = int(math.floor(a))
    m_f = (a - g) * 60.0
    m = int(math.floor(m_f))
    s_f = (m_f - m) * 60.0
    # arredonda segundos para inteiro
    s = int(round(s_f))
    # ajuste de 60s -> +1min
    if s >= 60:
        s = 0
        m += 1
    if m >= 60:
        m = 0
        g += 1
    return f"{sign}{g}°{m:02d}'{s:02d}\""

# -------------------- preparar dataframe estável (session_state) -----------
required_cols = [
    'EST','PV',
    'AnguloHorizontal_PD','AnguloHorizontal_PI',
    'AnguloZenital_PD','AnguloZenital_PI',
    'DistanciaInclinada_PD','DistanciaInclinada_PI'
]

if "stable_df" not in st.session_state:
    # começar com modelo + 4 linhas vazias
    st.session_state.stable_df = pd.concat([template_df, pd.DataFrame([{} for _ in range(4)])], ignore_index=True)

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            new_df = pd.read_csv(uploaded)
        else:
            new_df = pd.read_excel(uploaded)
        st.success(f"Arquivo '{uploaded.name}' carregado ({len(new_df)} linhas).")
        # normalize column names loosely (map common variations)
        new_df_columns = {c: c for c in new_df.columns}
        for c in list(new_df.columns):
            low = c.strip().lower()
            if "horizontal" in low or ("ang" in low and "h" in low):
                if "pd" in low: new_df_columns[c] = "AnguloHorizontal_PD"
                elif "pi" in low: new_df_columns[c] = "AnguloHorizontal_PI"
            if "zenit" in low or "zenital" in low:
                if "pd" in low: new_df_columns[c] = "AnguloZenital_PD"
                elif "pi" in low: new_df_columns[c] = "AnguloZenital_PI"
            if "dist" in low and ("di" in low or "inclin" in low):
                if "pd" in low: new_df_columns[c] = "DistanciaInclinada_PD"
                elif "pi" in low: new_df_columns[c] = "DistanciaInclinada_PI"
        new_df = new_df.rename(columns=new_df_columns)
        # ensure required columns exist
        for col in required_cols:
            if col not in new_df.columns:
                new_df[col] = ""
        st.session_state.stable_df = new_df[required_cols].copy()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

# use a stable copy for editor
df_for_editor = st.session_state.stable_df.copy()
for col in required_cols:
    if col not in df_for_editor.columns:
        df_for_editor[col] = ""

st.markdown("### Tabela (edite as células; adicione/remova linhas livremente)")

edited = st.data_editor(
    df_for_editor[required_cols],
    num_rows="dynamic",
    key="editor_topografia"
)

# save back to session_state to keep structure stable across reruns
st.session_state.stable_df = edited.copy()

# -------------------- cálculos por linha -----------------------------------
results = edited.copy()

# parse angles (AH and AZ) to decimal degrees
for c in ['AnguloHorizontal_PD','AnguloHorizontal_PI','AnguloZenital_PD','AnguloZenital_PI']:
    results[c + '_deg'] = results[c].apply(parse_angle_to_decimal)

# parse distances to numeric (m)
for c in ['DistanciaInclinada_PD','DistanciaInclinada_PI']:
    results[c + '_m'] = pd.to_numeric(results[c], errors='coerce')

# compute Dh for PD and PI using respective zenital angles:
# Dh_PD = DI_PD * sin(AZ_PD)
results['Dh_PD_m'] = results.apply(
    lambda r: (r['DistanciaInclinada_PD_m'] * math.sin(math.radians(r['AnguloZenital_PD_deg'])))
    if pd.notna(r.get('DistanciaInclinada_PD_m')) and pd.notna(r.get('AnguloZenital_PD_deg')) else np.nan,
    axis=1
)

results['Dh_PI_m'] = results.apply(
    lambda r: (r['DistanciaInclinada_PI_m'] * math.sin(math.radians(r['AnguloZenital_PI_deg'])))
    if pd.notna(r.get('DistanciaInclinada_PI_m')) and pd.notna(r.get('AnguloZenital_PI_deg')) else np.nan,
    axis=1
)

# compute AH mean (PD & PI) and convert to DMS
results['AH_med_deg'] = results[['AnguloHorizontal_PD_deg','AnguloHorizontal_PI_deg']].mean(axis=1, skipna=True)
results['AH_med_DMS'] = results['AH_med_deg'].apply(decimal_to_dms)

# -------------------- format output lines exactly as requested ----------------
lines = []
for _, r in results.iterrows():
    dh_pd = r.get('Dh_PD_m')
    dh_pi = r.get('Dh_PI_m')
    ah_dms = r.get('AH_med_DMS') or ""
    dh_pd_s = f"{dh_pd:.3f} m" if pd.notna(dh_pd) else ""
    dh_pi_s = f"{dh_pi:.3f} m" if pd.notna(dh_pi) else ""
    lines.append(f"{dh_pd_s}\t{dh_pi_s}\t{ah_dms}")

st.markdown("---")
st.subheader("Saída (cada linha: Dh_PD \t Dh_PI \t AH_médio DMS)")
st.code("\n".join(lines), language="text")

# opcional: tabela exibindo valores numéricos para conferência
show_table = st.checkbox("Mostrar tabela de conferência (valores numéricos)", value=False)
if show_table:
    display_df = results.copy()
    display_df['Dh_PD_m'] = display_df['Dh_PD_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    display_df['Dh_PI_m'] = display_df['Dh_PI_m'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    display_df['AH_med_DMS'] = display_df['AH_med_DMS'].fillna("")
    st.dataframe(display_df[[
        'EST','PV',
        'DistanciaInclinada_PD','DistanciaInclinada_PI',
        'AnguloZenital_PD','AnguloZenital_PI',
        'Dh_PD_m','Dh_PI_m','AH_med_DMS'
    ]], use_container_width=True)

# -------------------- permitir download da saída em CSV -----------------------
out_df = results[['Dh_PD_m','Dh_PI_m','AH_med_DMS']].copy()
out_df.rename(columns={'Dh_PD_m':'Dh_PD_m (m)','Dh_PI_m':'Dh_PI_m (m)','AH_med_DMS':'AH_med_DMS'}, inplace=True)
out_csv = out_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar saída (CSV)", data=out_csv, file_name="saida_topografia.csv", mime="text/csv")

st.markdown("---")
st.caption("Observação: para gerar/baixar o modelo Excel (.xlsx) no servidor, certifique-se de incluir 'openpyxl' no requirements.txt.")
