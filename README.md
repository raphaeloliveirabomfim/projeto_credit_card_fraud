# Detecção de Fraudes em Transações Bancárias

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

## Sobre o Projeto

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso de Ciência de Dados, simulando um case real de detecção de fraudes para o Banco Global Trust. O objetivo principal é construir um modelo de machine learning capaz de identificar transações fraudulentas com alta taxa de recall (prioridade do stakeholder), mesmo em um cenário de dados extremamente desbalanceados.

### O Desafio

- **Contexto:** Banco Global Trust - Gestão de Riscos
- **Prioridade explícita:** Minimizar FALSOS NEGATIVOS (fraudes não detectadas)
- **Dataset:** 284.807 transações de crédito (setembro/2023)
- **Fraudes:** 492 transações (apenas 0,172% do total)
- **Features:** 28 componentes PCA (anonimizadas) + Time + Amount

---

## Principais Resultados

| Métrica | Regressão Logística | LightGBM Otimizado |
|---------|---------------------|--------------------|
| **Recall** | **92,0%** | 86,7% |
| **Falsos Negativos** | **6** | 10 |
| Falsos Positivos | 2.683 | 1.280 |
| Precisão | 2,51% | 4,83% |
| F1-Score | 0,0488 | 0,0915 |
| PR-AUC | 0,79 | 0,79 |

**Modelo Final Selecionado:** Regressão Logística com class_weight (atende à prioridade do stakeholder)

---

## Tecnologias Utilizadas

| Categoria | Tecnologias |
|-----------|-------------|
| **Linguagem** | Python 3.8+ |
| **Análise de Dados** | Pandas, NumPy |
| **Visualização** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, Imbalanced-learn |
| **Otimização** | RandomizedSearchCV |
| **Ambiente** | Jupyter Notebook |

---

## Metodologia

### 1. Análise Exploratória (EDA)
- Distribuição temporal das fraudes
- Análise da feature `Amount` (valores das transações)
- Visualização com t-SNE (identificação de clusters)
- Correlação entre componentes PCA (ortogonalidade)

### 2. Pré-processamento
- Transformação log em `Amount`
- Features temporais
- Divisão **TEMPORAL** dos dados 
- Padronização com StandardScaler

### 3. Abordagens para Desbalanceamento
- Baseline com `class_weight`
- Undersampling
- SMOTE (oversampling)
- SMOTE + Tomek (híbrido)

### 4. Modelagem
- Regressão Logística (baseline)
- Random Forest
- XGBoost
- LightGBM (com ajuste fino de hiperparâmetros)

### 5. Otimização
- RandomizedSearchCV com 50 iterações
- Validação cruzada estratificada (3 folds)
- Foco em **RECALL** (prioridade do stakeholder)

---

## 💡 Principais Insights

1. **Fraudes formam clusters** - A visualização t-SNE mostrou que transações fraudulentas tendem a se agrupar, indicando padrões distintos.

2. **Componentes PCA são ortogonais** - Correlação zero entre as features, eliminando preocupações com multicolinearidade.

3. **Abordagem mais eficaz** - Baseline com `class_weight` superou técnicas mais complexas como SMOTE e undersampling.

4. **Trade-off claro** - A Regressão Logística oferece o maior recall, mas ao custo de mais falsos positivos.

5. **Valor de negócio** - O modelo evita R$ 41.400 em prejuízos no período de teste (~R$ 500k/ano).

---

## Limitações do Modelo

- **Falsos positivos elevados** (2.683 clientes legítimos bloqueados)
- **Falta de interpretabilidade** (features PCA anonimizadas)
- **Dados estáticos** (treinado apenas com setembro/2023)
- **Threshold fixo** (não considera o valor da transação)

---

##  Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## Autor

**Raphael Oliveira Bomfim**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raphael-oliveira-bomfim-73b808260/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://https://github.com/raphaeloliveirabomfim/)

---

## Referências

- Dataset: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- SMOTE: [Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- Scikit-learn: [Machine Learning in Python](https://scikit-learn.org/)
- LightGBM: [Light Gradient Boosting Machine](https://lightgbm.readthedocs.io/)

---
