# 📐 Calculadora Topográfica — Estação Total

Aplicação web em **Streamlit** para cálculo de distâncias horizontais e ângulo horizontal médio a partir de leituras **PD / PI** de uma Estação Total.

## 🔧 Funcionalidades

- Upload de planilhas **Excel/CSV** com leituras de campo;
- Edição interativa dos dados diretamente na tabela da aplicação;
- Aceita ângulos:
  - em **DMS**: `235°47'33"`, `235 47 33`, `235:47:33`;
  - em **graus decimais**: `235.7925`;
- Cálculo automático de:
  - `Dh_PD (m)` — distância horizontal lado PD;
  - `Dh_PI (m)` — distância horizontal lado PI;
  - `AH_médio (DMS)` — ângulo horizontal médio entre PD e PI;
- Download:
  - Modelo de planilha (`modelo_estacao_total.xlsx`);
  - Saída com resultados em `saida_topografia.csv`;
- Layout escuro com CSS customizado.

## 🧮 Fórmulas utilizadas

- Conversão de ângulos DMS → decimal;
- Distância horizontal:
  
  $$Dh = DI \\cdot \\sin(AZ)$$

  onde:
  - \( DI \) = distância inclinada (m);
  - \( AZ \) = ângulo zenital em graus decimais.

- Ângulo horizontal médio:

  $$ AH_{médio} = \\dfrac{AH_{PD} + AH_{PI}}{2} $$

## ▶️ Como executar

1. Crie e ative um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\\Scripts\\activate   # Windows

# Calculadora de Ângulos e Distâncias – Método das Direções (UFPE)

Este projeto implementa, em um único arquivo Python (`app_unico.py`), uma aplicação interativa desenvolvida com **Streamlit** para processamento de observações de estação total pelo **método das direções**, com foco em:

- Cálculo de direções horizontais médias (Hz).
- Cálculo de ângulos verticais / zenitais corrigidos.
- Cálculo de distâncias horizontais médias (DH).
- Construção e análise geométrica do **triângulo** formado pelos pontos **P1, P2, P3**.
- Geração de tabelas consolidadas para uso didático em disciplinas de Topografia / Equipamentos de Medição (UFPE).
- Geração de um XLSX de saída com **resumo numérico + figura do triângulo**.

Toda a aplicação está escrita em **Python 3**, usando as seguintes bibliotecas:

- **Streamlit** (UI / interface web).
- **pandas** (manipulação de dados tabulares).
- **numpy** (cálculos numéricos).
- **matplotlib** (plotagem do triângulo).
- **XlsxWriter** (geração de arquivos `.xlsx`).
- **openpyxl** (leitura de planilhas Excel `.xlsx`).
- **python-dateutil** (parsing de datas – indireto via `pandas`).

---

## 1. Arquitetura Lógica do Aplicativo (em um único arquivo)

Embora todo o código esteja concentrado em um único arquivo `app_unico.py`, ele é logicamente dividido em **módulos internos**:

1. **Funções auxiliares de identificação**  
   - Responsáveis por ler a aba **“Identificação”** da planilha Excel e extrair:
     - `Professor(a)`
     - `Equipamento`
     - `Dados` (data da atividade)
     - `Local`
     - `Patrimônio`
   - Linguagem: **Python 3** puro, com uso de `pandas` e `datetime`.

2. **Funções de processamento**  
   - Conjunto de funções que tratam:
     - Parsing de ângulos em graus, minutos e segundos (GMS) para decimal.
     - Conversão de decimal para string GMS.
     - Validação de colunas e linhas da aba **“Dados”** da planilha.
     - Cálculos de Hz, Z corrigido, DH, DN, médias de séries, etc.
     - Cálculo do triângulo (lados, ângulos internos, área).
   - Linguagem: **Python 3** + **numpy** + **pandas**.

3. **Funções de plotagem e geração de XLSX de resultado**  
   - Geração de figura 2D (planta) do triângulo.
   - Exportação de um arquivo `.xlsx` contendo:
     - Uma aba com o resumo dos comprimentos e ângulos.
     - Uma aba com a figura do triângulo inserida como imagem.
   - Linguagem: **Python 3**, **matplotlib**, **pandas**, **XlsxWriter**.

4. **Interface usuário (UI) com Streamlit**  
   - Organização em duas “páginas lógicas” via `st.session_state`:
     - Página **Carregar dados**.
     - Página **Processamento**.
   - Estilização detalhada com CSS injetado diretamente na aplicação.
   - Linguagem: **Python 3** + **Streamlit** + CSS (incorporado como string no código).

---

## 2. Fluxo Funcional da Aplicação

### 2.1. Modelo de Planilha

Na **Página 1 – Carregar dados**, o usuário pode baixar um **modelo padrão** de Excel gerado dinamicamente pela função:

```python
gerar_modelo_excel_bytes()  # Python + pandas + XlsxWriter
