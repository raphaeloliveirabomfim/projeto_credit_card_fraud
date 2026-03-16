# Detecção de Fraudes em Cartão de Crédito

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Visão Geral

Modelo preditivo para **detecção em tempo real de transações fraudulentas** em cartões de crédito, desenvolvido com foco em maximizar o **Recall** — cada fraude não detectada representa prejuízo financeiro direto ao banco e ao cliente.

> **Contexto:** O Banco Global Trust registra 284.807 transações em 48 horas, das quais apenas **492 são fraudes (0,172%)**. O desbalanceamento extremo é o principal desafio técnico deste projeto.

---

## Problema de Negócio

| Tipo de Erro | Consequência | Custo |
|---|---|---|
| **Falso Negativo** (fraude não detectada) | Prejuízo financeiro direto | Alto |
| **Falso Positivo** (legítima bloqueada) | Experiência negativa do cliente | Médio |

**Decisão:** Maximizar Recall usando F2-Score (Recall com peso 2×) para calibrar o threshold.

---

## Estrutura do Projeto

```
credit-card-fraud/
│
├── notebooks/
│   ├── 01_eda.ipynb           # Análise exploratória completa
│   ├── 02_modelagem.ipynb     # Pipeline, modelos e avaliação
│   └── 03_interpretacao.ipynb # Impacto financeiro e análise de negócio
│
├── data/
│   ├── creditcard.csv         # Dataset (baixar do Kaggle — ver data/README.md)
│   └── README.md              # Instruções de download e dicionário de dados
│
├── figures/                 # Gráficos gerados automaticamente
│
├── models/
│   ├── modelo_final.pkl        # Modelo salvo (pipeline)
│   ├── scaler.pkl              # RobustScaler ajustado no treino
│   ├── threshold_final.pkl     # Threshold otimizado por F2-Score
│   ├── features.pkl            # Lista de features usadas
│   └── resultados_modelos.csv  # Tabela comparativa de métricas
│
├── requirements.txt
└── README.md
```

---

## Dados

Dataset público do Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Coluna | Descrição |
|--------|-----------|
| `V1`–`V28` | Componentes PCA (features anonimizadas) |
| `Time` | Segundos desde a primeira transação |
| `Amount` | Valor da transação |
| `Class` | **Target** — 0: legítima, 1: fraude |

---

## Metodologia

### Diferenciais técnicos

- **Divisão temporal** (em vez de aleatória) para evitar data leakage em séries temporais de transações
- **RobustScaler** em vez de StandardScaler — resistente aos outliers significativos do dataset
- **Features cíclicas** de tempo (sin/cos) para capturar padrão de hora do dia
- **Threshold otimizado por F2-Score** em vez do default 0.5
- **Precision-Recall Curve** como métrica principal (mais informativa que ROC para classes desbalanceadas)

### Pipeline

```
Dados - Feature Engineering - Divisão Temporal (80/20)
      - RobustScaler → SMOTE (apenas treino)
      - Modelos: LR (baseline), Random Forest, LightGBM
      - Threshold F2-Score → Avaliação Final
```

---

## Resultados

| Modelo | Threshold | Recall | PR-AUC | F2-Score | FN |
|--------|-----------|--------|--------|----------|----|
| Regressão Logística | 1.00 | 0.786 | 0.776 | 0.772 | 16 |
| Random Forest | 0.548 | 0.773 | 0.777 | 0.792 | 17 |
| LightGBM | 0.969 | 0.7467 | 0.777 | 0.779 | 19 |

> *Execute os notebooks para ver os valores reais após rodar com o dataset.*

---

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/credit-card-fraud.git
cd credit-card-fraud

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o dataset (ver data/README.md) e coloque em data/creditcard.csv

# 4. Execute os notebooks em ordem
jupyter notebook notebooks/01.eda.ipynb
jupyter notebook notebooks/02.modelagem.ipynb
jupyter notebook notebooks/03.interpretacao.ipynb
```

---

## Autor

**Raphael Oliveira Bomfim**
- LinkedIn: [https://www.linkedin.com/in/raphael-oliveira-bomfim/](https://linkedin.com)
- GitHub: [https://github.com/raphaeloliveirabomfim](https://github.com)
- Email: rapha.sep@hotmail.com
  
Desenvolvido como projeto de portfólio de Ciência de Dados.

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
