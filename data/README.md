# 📊 Dados — Credit Card Fraud Detection

## Como obter o dataset

### Opção 1 — Kaggle (recomendado)
1. Acesse: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Clique em **Download**
3. Extraia e coloque o arquivo `creditcard.csv` nesta pasta (`data/`)

### Opção 2 — Kaggle API
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

---

## Dicionário de Dados

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Time` | float | Segundos desde a primeira transação do dataset |
| `V1`–`V28` | float | Componentes PCA — features originais anonimizadas por privacidade |
| `Amount` | float | Valor da transação em euros |
| `Class` | int | **Target** — 0: legítima, 1: fraude |

## Características do Dataset

| Informação | Valor |
|-----------|-------|
| Total de transações | 284.807 |
| Fraudes | 492 (0,172%) |
| Período coberto | ~48 horas |
| Origem | Cartões europeus, setembro de 2013 |
| Features originais | Anonimizadas via PCA |

## Referência

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.  
> *Calibrating Probability with Undersampling for Unbalanced Classification.*  
> IEEE SSCI 2015.
