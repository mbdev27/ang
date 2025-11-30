# app.py
# Média das Direções (Hz) + Z + DH/DN - UFPE (versão arquivo único)

import io
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ==================== Config página ====================
st.set_page_config(
    page_title="Média das Direções (Hz) — Estação Total | UFPE",
    layout="wide",
    page_icon="📐",
)

# ==================== Funções auxiliares ====================

REQUIRED_COLS = ["EST", "PV", "Hz_PD", "Hz_PI", "Z_PD", "Z_PI", "DI_PD", "DI_PI"]


def parse_angle_to_decimal(value: str) -> float:
    """
    Converte string de ângulo em DMS (145°47′33″, 145°47'33", 145 47 33)
    ou decimal ("145.7925") para graus decimais.
    Retorna NaN se não conseguir converter.
    """
    if value is None:
        return float("nan")

    s = str(value).strip()
    if s == "":
        return float("nan")

    # 1) tenta decimal simples
    try:
        if all(ch.isdigit() or ch in ".,-+" for ch in s):
            return float(s.replace(",", "."))
    except Exception:
        pass

    # 2) normaliza símbolos DMS para espaços
    for ch in ["°", "º", "'", "’", "´", "′", '"', "″"]:
        s = s.replace(ch, " ")

    # vírgula como ponto
    s = s.replace(",", ".")

    parts = s.split()
    parts = [p for p in parts if p != ""]
    if len(parts) == 0:
        return float("nan")

    # 3) interpreta como D, M, S
    try:
        deg = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
    except Exception:
        return float("nan")

    sign = 1.0
    if deg < 0:
        sign = -1.0
        deg = abs(deg)

    return sign * (deg + minutes / 60.0 + seconds / 3600.0)


def decimal_to_dms(angle_deg: float) -> str:
    """
    Converte graus decimais para string DMS com segundos inteiros: 145°47'34"
    """
    if angle_deg is None or math.isnan(angle_deg):
        return ""
    sign = "-" if angle_deg < 0 else ""
    a = abs(angle_deg)

    d = int(a)
    m_f = (a - d) * 60
    m = int(m_f)
    s_f = (m_f - m) * 60

    # arredonda segundos para inteiro
    s = int(round(s_f))

    # ajusta “estouro” de 60"
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d += 1

    return f"{sign}{d:02d}°{m:02d}'{s:02d}\""


def mean_direction_two(a_deg: float, b_deg: float) -> float:
    """
    Média vetorial de duas direções em graus.
    """
    if math.isnan(a_deg) or math.isnan(b_deg):
        return float("nan")
    a_rad = math.radians(a_deg)
    b_rad = math.radians(b_deg)
    x = math.cos(a_rad) + math.cos(b_rad)
    y = math.sin(a_rad) + math.sin(b_rad)
    if x == 0 and y == 0:
        return float("nan")
    ang = math.degrees(math.atan2(y, x))
    if ang < 0:
        ang += 360.0
    return ang


def mean_direction_list(angles_deg: pd.Series) -> float:
    """
    Média vetorial de uma lista (Series) de ângulos em graus.
    """
    vals = [a for a in angles_deg if not math.isnan(a)]
    if len(vals) == 0:
        return float("nan")
    x = sum(math.cos(math.radians(v)) for v in vals)
    y = sum(math.sin(math.radians(v)) for v in vals)
    if x == 0 and y == 0:
        return float("nan")
    ang = math.degrees(math.atan2(y, x))
    if ang < 0:
        ang += 360.0
    return ang


def normalizar_colunas(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Harmoniza nomes de colunas vindos de planilhas diversas para:
    EST, PV, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI.
    """
    df = df_original.copy()
    colmap = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ["est", "estacao", "estação"]:
            colmap[c] = "EST"
        elif low in ["pv", "ponto visado", "ponto_visado", "ponto"]:
            colmap[c] = "PV"
        elif ("horizontal" in low and "pd" in low) or ("hz" in low and "pd" in low):
            colmap[c] = "Hz_PD"
        elif ("horizontal" in low and "pi" in low) or ("hz" in low and "pi" in low):
            colmap[c] = "Hz_PI"
        elif ("zenital" in low and "pd" in low) or ("z" in low and "pd" in low):
            colmap[c] = "Z_PD"
        elif ("zenital" in low and "pi" in low) or ("z" in low and "pi" in low):
            colmap[c] = "Z_PI"
        elif "dist" in low and "pd" in low:
            colmap[c] = "DI_PD"
        elif "dist" in low and "pi" in low:
            colmap[c] = "DI_PI"
        else:
            colmap[c] = c
    return df.rename(columns=colmap)


def validar_dataframe(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normaliza colunas e verifica:
      - Presença de colunas obrigatórias.
      - Se Hz/Z/DI são conversíveis para ângulo/float.
    Retorna (df_normalizado, lista_de_erros).
    """
    erros: List[str] = []
    df = normalizar_colunas(df_original)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        erros.append("Colunas obrigatórias ausentes: " + ", ".join(missing))

    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""

    invalid_rows_hz: List[int] = []
    invalid_rows_z: List[int] = []
    invalid_rows_di: List[int] = []

    for idx, row in df.iterrows():
        hz_pd = parse_angle_to_decimal(row.get("Hz_PD", ""))
        hz_pi = parse_angle_to_decimal(row.get("Hz_PI", ""))
        z_pd = parse_angle_to_decimal(row.get("Z_PD", ""))
        z_pi = parse_angle_to_decimal(row.get("Z_PI", ""))
        if np.isnan(hz_pd) or np.isnan(hz_pi):
            invalid_rows_hz.append(idx + 1)
        if np.isnan(z_pd) or np.isnan(z_pi):
            invalid_rows_z.append(idx + 1)
        try:
            di_pd = float(str(row.get("DI_PD", "")).replace(",", "."))
            di_pi = float(str(row.get("DI_PI", "")).replace(",", "."))
            if np.isnan(di_pd) or np.isnan(di_pi):
                invalid_rows_di.append(idx + 1)
        except Exception:
            invalid_rows_di.append(idx + 1)

    if invalid_rows_hz:
        erros.append(
            "Valores inválidos ou vazios em Hz_PD / Hz_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_hz))
            + "."
        )
    if invalid_rows_z:
        erros.append(
            "Valores inválidos ou vazios em Z_PD / Z_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_z))
            + "."
        )
    if invalid_rows_di:
        erros.append(
            "Valores inválidos ou vazios em DI_PD / DI_PI nas linhas: "
            + ", ".join(map(str, invalid_rows_di))
            + "."
        )

    return df, erros


def calcular_linha_a_linha(df_uso: pd.DataFrame) -> pd.DataFrame:
    """
    Converte ângulos, distâncias, e calcula Hz_médio, DH/DN linha a linha.
    """
    res = df_uso.copy()

    # Ângulos em decimal
    for col in ["Hz_PD", "Hz_PI", "Z_PD", "Z_PI"]:
        res[col + "_deg"] = res[col].apply(parse_angle_to_decimal)

    # Distâncias inclinadas
    res["DI_PD_m"] = res["DI_PD"].apply(lambda x: float(str(x).replace(",", ".")))
    res["DI_PI_m"] = res["DI_PI"].apply(lambda x: float(str(x).replace(",", ".")))

    z_pd_rad = res["Z_PD_deg"] * np.pi / 180.0
    z_pi_rad = res["Z_PI_deg"] * np.pi / 180.0

    # DH / DN (3 casas decimais)
    res["DH_PD_m"] = np.abs(res["DI_PD_m"] * np.sin(z_pd_rad)).round(3)
    res["DN_PD_m"] = np.abs(res["DI_PD_m"] * np.cos(z_pd_rad)).round(3)
    res["DH_PI_m"] = np.abs(res["DI_PI_m"] * np.sin(z_pi_rad)).round(3)
    res["DN_PI_m"] = np.abs(res["DI_PI_m"] * np.cos(z_pi_rad)).round(3)

    # Hz médio linha a linha
    res["Hz_med_deg"] = res.apply(
        lambda r: mean_direction_two(r["Hz_PD_deg"], r["Hz_PI_deg"]), axis=1
    )
    res["Hz_med_DMS"] = res["Hz_med_deg"].apply(decimal_to_dms)

    # DH/DN médios linha a linha (3 casas)
    res["DH_med_m"] = np.abs((res["DH_PD_m"] + res["DH_PI_m"]) / 2.0).round(3)
    res["DN_med_m"] = np.abs((res["DN_PD_m"] + res["DN_PI_m"]) / 2.0).round(3)

    return res


def agregar_por_par(res: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega linha a linha em um DataFrame por par EST–PV:
      - Hz/Z médios com média vetorial.
      - DI médias aritméticas.
      - DH/DN médios derivados.
    """

    def agg_par(df_group: pd.DataFrame) -> pd.Series:
        out = {}
        out["Hz_PD_med_deg"] = mean_direction_list(df_group["Hz_PD_deg"])
        out["Hz_PI_med_deg"] = mean_direction_list(df_group["Hz_PI_deg"])
        out["Z_PD_med_deg"] = mean_direction_list(df_group["Z_PD_deg"])
        out["Z_PI_med_deg"] = mean_direction_list(df_group["Z_PI_deg"])
        out["DI_PD_med_m"] = float(df_group["DI_PD_m"].mean())
        out["DI_PI_med_m"] = float(df_group["DI_PI_m"].mean())
        return pd.Series(out)

    df_par = res.groupby(["EST", "PV"], as_index=False).apply(agg_par)

    # Hz médio por par
    df_par["Hz_med_deg_par"] = df_par.apply(
        lambda r: mean_direction_two(r["Hz_PD_med_deg"], r["Hz_PI_med_deg"]), axis=1
    )
    df_par["Hz_med_DMS_par"] = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    # DH/DN médios por par (3 casas)
    zpd_par_rad = df_par["Z_PD_med_deg"] * np.pi / 180.0
    zpi_par_rad = df_par["Z_PI_med_deg"] * np.pi / 180.0

    df_par["DH_PD_m_par"] = np.abs(
        df_par["DI_PD_med_m"] * np.sin(zpd_par_rad)
    ).round(3)
    df_par["DN_PD_m_par"] = np.abs(
        df_par["DI_PD_med_m"] * np.cos(zpd_par_rad)
    ).round(3)
    df_par["DH_PI_m_par"] = np.abs(
        df_par["DI_PI_med_m"] * np.sin(zpi_par_rad)
    ).round(3)
    df_par["DN_PI_m_par"] = np.abs(
        df_par["DI_PI_med_m"] * np.cos(zpi_par_rad)
    ).round(3)

    df_par["DH_med_m_par"] = np.abs(
        (df_par["DH_PD_m_par"] + df_par["DH_PI_m_par"]) / 2.0
    ).round(3)
    df_par["DN_med_m_par"] = np.abs(
        (df_par["DN_PD_m_par"] + df_par["DN_PI_m_par"]) / 2.0
    ).round(3)

    return df_par


def tabela_medicao_angular_horizontal(df_par: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela no formato do slide:
    'Medição Angular Horizontal'
    Colunas: EST, PV, Hz PD, Hz PI, Hz Médio, Hz Reduzido, Média das Séries.
    """
    hz_pd_med_dms = df_par["Hz_PD_med_deg"].apply(decimal_to_dms)
    hz_pi_med_dms = df_par["Hz_PI_med_deg"].apply(decimal_to_dms)
    hz_med_dms = df_par["Hz_med_deg_par"].apply(decimal_to_dms)

    tab = pd.DataFrame(
        {
            "EST": df_par["EST"],
            "PV": df_par["PV"],
            "Hz PD": hz_pd_med_dms,
            "Hz PI": hz_pi_med_dms,
            "Hz Médio": hz_med_dms,
            "Hz Reduzido": hz_med_dms,       # depois podemos aplicar Ré/Vante aqui
            "Média das Séries": hz_med_dms,  # média das séries PD/PI
        }
    )
    return tab


def tabela_medicao_angular_vertical(df_par: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela no formato do slide:
    'Medição Angular Vertical/Zenital'
    Colunas: EST, PV, Z PD, Z PI, Z Corrigido, Média das Séries.
    Usa: Z = (Z_PD_med - Z_PI_med) / 2 + 180°
    """
    z_pd_med = df_par["Z_PD_med_deg"]
    z_pi_med = df_par["Z_PI_med_deg"]

    z_corr_deg = (z_pd_med - z_pi_med) / 2.0 + 180.0

    z_pd_med_dms = z_pd_med.apply(decimal_to_dms)
    z_pi_med_dms = z_pi_med.apply(decimal_to_dms)
    z_corr_dms = z_corr_deg.apply(decimal_to_dms)

    tab = pd.DataFrame(
        {
            "EST": df_par["EST"],
            "PV": df_par["PV"],
            "Z PD": z_pd_med_dms,
            "Z PI": z_pi_med_dms,
            "Z Corrigido": z_corr_dms,
            "Média das Séries": z_corr_dms,
        }
    )
    return tab


# ==================== CSS e layout visual UFPE ====================

CUSTOM_CSS = """
<style>
body, .stApp {
    background: radial-gradient(circle at top left, #faf5f5 0%, #f7f5f5 45%, #f4f4f4 100%);
    color: #111827;
    font-family: "Trebuchet MS", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
.main-card {
    background: linear-gradient(145deg, #ffffff 0%, #fdfbfb 40%, #ffffff 100%);
    border-radius: 18px;
    padding: 1.8rem 2.1rem;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 18px 40px rgba(15, 23, 42, 0.22),
        0 0 0 1px rgba(15, 23, 42, 0.03);
}
.ufpe-top-bar {
    width: 100%;
    height: 8px;
    border-radius: 0 0 14px 14px;
    background: linear-gradient(90deg, #5b0000 0%, #990000 52%, #5b0000 100%);
    margin-bottom: 0.9rem;
}
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
.app-subtitle {
    font-size: 0.95rem;
    color: #4b5563;
    margin-bottom: 0.9rem;
}
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
    background: radial-gradient(circle at 30% 30%, #ffffff 0%, #990000 40%, #111827 100%);
}
.helper-box {
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    background: linear-gradient(135deg, #fdf2f2 0%, #fff5f5 40%, #fdf2f2 100%);
    border: 1px solid rgba(153, 0, 0, 0.25);
    font-size: 0.83rem;
    color: #4b5563;
    margin-bottom: 0.7rem;
}
.footer-text {
    font-size: 0.75rem;
    color: #6b7280;
}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background: linear-gradient(145deg, #ffffff 0%, #f9fafb 60%, #ffffff 100%) !important;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
}
[data-testid="stDataFrame"] thead tr {
    background: linear-gradient(90deg, #f5e6e8 0%, #fdf2f2 100%) !important;
    color: #5b101d !important;
    font-weight: 600;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: #fafafa !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #f3f0f0 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def cabecalho_ufpe():
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown('<div class="ufpe-top-bar"></div>', unsafe_allow_html=True)

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

        st.markdown('<hr class="ufpe-separator">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="app-title">
                <span class="icon">📐</span>
                <span>Média das Direções (Hz) — Estação Total</span>
            </div>
            <div class="app-subtitle">
                Cálculo da média das direções Hz e do ângulo vertical (Z) por séries PD/PI.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="helper-box">
                <b>Modelo esperado de planilha:</b><br>
                Colunas: <code>EST</code>, <code>PV</code>,
                <code>Hz_PD</code>, <code>Hz_PI</code>,
                <code>Z_PD</code>, <code>Z_PI</code>,
                <code>DI_PD</code>, <code>DI_PI</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def secao_modelo_e_upload():
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>1. Modelo de dados (Hz, Z e DI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_df = pd.DataFrame(
        {
            "EST": ["A", "A"],
            "PV": ["B", "C"],
            "Hz_PD": ["00°00'00\"", "18°58'22\""],
            "Hz_PI": ["179°59'48\"", "198°58'14\""],
            "Z_PD": ["90°51'08\"", "90°51'25\""],
            "Z_PI": ["269°08'52\"", "269°08'33\""],
            "DI_PD": [10.0, 12.0],
            "DI_PI": [10.0, 12.0],
        }
    )

    excel_bytes = io.BytesIO()
    template_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)
    st.download_button(
        "📥 Baixar modelo Excel (.xlsx)",
        data=excel_bytes.getvalue(),
        file_name="modelo_medicao_direcoes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

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
        "Envie a planilha preenchida (EST, PV, Hz_PD, Hz_PI, Z_PD, Z_PI, DI_PD, DI_PI)",
        type=["xlsx", "xls", "csv"],
    )
    return uploaded


def processar_upload(uploaded):
    if uploaded is None:
        return None

    try:
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

    st.success(f"Arquivo '{uploaded.name}' carregado ({len(raw_df)} linhas).")

    df_valid, erros = validar_dataframe(raw_df)
    st.subheader("Pré-visualização dos dados importados")
    st.dataframe(df_valid[REQUIRED_COLS], use_container_width=True)

    if erros:
        st.error("Não foi possível calcular diretamente devido aos seguintes problemas:")
        for e in erros:
            st.markdown(f"- {e}")
        return None
    else:
        return df_valid[REQUIRED_COLS].copy()


def secao_calculos(df_uso: pd.DataFrame):
    st.markdown(
        """
        <div class="section-title">
            <span class="dot"></span>
            <span>3. Medições Angulares Horizontal e Vertical (médias por par)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Linha a linha
    res = calcular_linha_a_linha(df_uso)

    st.markdown("##### Tabela linha a linha (cada série PD/PI)")
    cols_linha = [
        "EST",
        "PV",
        "Hz_PD",
        "Hz_PI",
        "Hz_med_DMS",
        "Z_PD",
        "Z_PI",
        "DH_PD_m",
        "DH_PI_m",
        "DH_med_m",
    ]
    df_linha = res[cols_linha].copy()

    # Formata DH com vírgula e 3 casas decimais
    for c in ["DH_PD_m", "DH_PI_m", "DH_med_m"]:
        df_linha[c] = df_linha[c].apply(
            lambda x: f"{x:.3f}".replace(".", ",") if pd.notna(x) else ""
        )

    st.dataframe(df_linha, use_container_width=True)

    # Agregado por par EST–PV
    df_par = agregar_por_par(res)

    # Tabela Horizontal
    st.markdown("##### Medição Angular Horizontal")
    st.markdown(
        r"""
**Fórmulas utilizadas (Hz médio e Hz reduzido)**  

Média das direções (por série PD/PI):  

\[
Hz = \frac{Hz_{PD} + Hz_{PI}}{2} \pm 90^\circ
\]

com:
- **+** se \(Hz_{PD} > Hz_{PI}\)  
- **−** se \(Hz_{PD} < Hz_{PI}\)

Cálculo do ângulo entre duas direções (redução entre Ré e Vante):  

\[
\alpha = Hz_{\text{Vante}} - Hz_{\text{Ré}}
\]
""",
        unsafe_allow_html=False,
    )
    tab_hz = tabela_medicao_angular_horizontal(df_par)
    st.dataframe(tab_hz, use_container_width=True)

    # Tabela Vertical
    st.markdown("##### Medição Angular Vertical/Zenital")
    st.markdown(
        r"""
**Fórmula utilizada (Z corrigido)**  

\[
Z = \frac{Z'_{PD} - Z'_{PI}}{2} + 180^\circ
\]
""",
        unsafe_allow_html=False,
    )
    tab_z = tabela_medicao_angular_vertical(df_par)
    st.dataframe(tab_z, use_container_width=True)


def rodape():
    st.markdown(
        """
        <p class="footer-text">
            Versão do app: <code>único_arquivo_1.1 — Tabelas no formato do slide (Hz / Z), DMS com segundos inteiros, DH com 3 casas.</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # fecha main-card


# ==================== Fluxo principal ====================

cabecalho_ufpe()
uploaded = secao_modelo_e_upload()
df_uso = processar_upload(uploaded)

if df_uso is not None:
    secao_calculos(df_uso)

rodape()
