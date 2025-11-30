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
