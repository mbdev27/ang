import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Cálculo de Estação Total", layout="wide")

st.title("📡 Calculadora de Ângulos e Distâncias (Estação Total)")

# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

def dms_to_deg(dms_str):
    try:
        d, m, s = dms_str.replace("°", " ").replace("'", " ").replace('"', " ").split()
        return float(d) + float(m)/60 + float(s)/3600
    except:
        return np.nan

def deg_to_dms(deg):
    if np.isnan(deg):
        return ""
    d = int(deg)
    m_float = abs(deg - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return f"{d}°{m:02d}'{s:02.0f}\""

def calcular_dist_h(dist_inclinada, ang_zenital_deg):
    ang_rad = np.radians(ang_zenital_deg)
    return dist_inclinada * np.sin(ang_rad)

# ---------------------------------------------------------
# Tabela inicial
# ---------------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "EST": [""],
        "PV": [""],
        "AH_PD": [""],
        "AH_PI": [""],
        "AZ_PD": [""],
        "AZ_PI": [""],
        "DI_PD": [""],
        "DI_PI": [""],
    })

st.subheader("📝 Tabela de Entrada (edite diretamente)")

df_edit = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    key="editor"
)

st.session_state.df = df_edit

# ---------------------------------------------------------
# Quando clicar em calcular
# ---------------------------------------------------------

if st.button("🔍 Calcular"):
    df = st.session_state.df.copy()

    # Conversões
    df["AH_PD_deg"] = df["AH_PD"].apply(dms_to_deg)
    df["AH_PI_deg"] = df["AH_PI"].apply(dms_to_deg)
    df["AZ_PD_deg"] = df["AZ_PD"].apply(dms_to_deg)
    df["AZ_PI_deg"] = df["AZ_PI"].apply(dms_to_deg)

    # Médias
    df["AH_med_deg"] = (df["AH_PD_deg"] + df["AH_PI_deg"]) / 2
    df["AH_med_DMS"] = df["AH_med_deg"].apply(deg_to_dms)

    # Dist horizontais
    df["DH_PD"] = df.apply(lambda x: calcular_dist_h(x["DI_PD"], x["AZ_PD_deg"]), axis=1)
    df["DH_PI"] = df.apply(lambda x: calcular_dist_h(x["DI_PI"], x["AZ_PI_deg"]), axis=1)

    # Dist horizontais formatadas
    df["DH_PD_fmt"] = df["DH_PD"].apply(lambda x: f"{x:.3f} m" if not np.isnan(x) else "")
    df["DH_PI_fmt"] = df["DH_PI"].apply(lambda x: f"{x:.3f} m" if not np.isnan(x) else "")

    # -----------------------------------------------------
    # Tabela final
    # -----------------------------------------------------
    st.subheader("📊 Resultados")

    resultado = pd.DataFrame({
        "EST": df["EST"],
        "PV": df["PV"],
        "DH_PD": df["DH_PD_fmt"],
        "DH_PI": df["DH_PI_fmt"],
        "AH_médio": df["AH_med_DMS"]
    })

    st.dataframe(resultado, use_container_width=True)

