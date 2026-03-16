"""
💳 Dashboard — Detecção de Fraudes em Cartão de Crédito
========================================================
Execute com:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve,
    roc_curve, roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, fbeta_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection · Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

COR_FRAUDE  = "#E74C3C"
COR_LEGIT   = "#27AE60"
COR_ACCENT  = "#2980B9"
COR_WARN    = "#F39C12"
COR_DARK    = "#1C2833"

TEMPLATE = dict(
    font=dict(family="IBM Plex Mono, monospace", color="#2C3E50"),
    margin=dict(t=40, b=30, l=20, r=20),
    title_font=dict(color="#1C2833", size=13),
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp { background-color: #0D1117; }

[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] > div * { color: #C9D1D9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #58A6FF !important; }

[data-testid="stTabs"] button p {
    color: #8B949E !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #58A6FF !important;
}
[data-testid="stTabs"] {
    background: transparent;
    border-bottom: 1px solid #30363D;
}

.main h1, .main h2, .main h3 { color: #E6EDF3 !important; }
.main p, .main span { color: #C9D1D9; }

/* Cards KPI */
.kpi-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 8px;
    border-left: 3px solid #58A6FF;
}
.kpi-card.danger  { border-left-color: #F85149; }
.kpi-card.success { border-left-color: #3FB950; }
.kpi-card.warning { border-left-color: #D29922; }

.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem; font-weight: 500;
    color: #8B949E; text-transform: uppercase;
    letter-spacing: 0.10em;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem; font-weight: 600;
    color: #E6EDF3; line-height: 1.2; margin: 4px 0;
}
.kpi-delta { font-size: 0.80rem; color: #8B949E; }

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.80rem; font-weight: 600;
    color: #58A6FF; text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0 0 8px 0; margin-top: 8px;
    border-bottom: 1px solid #30363D;
}

.alert-box {
    background: #1C1E26;
    border: 1px solid #F85149;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #F85149;
}
.alert-box.ok {
    border-color: #3FB950;
    color: #3FB950;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DADOS E MODELO
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

@st.cache_data
def carregar_dados():
    df = pd.read_csv(BASE / "data" / "creditcard.csv")
    df["Amount_Log"]       = np.log1p(df["Amount"])
    df["Time_Seconds_Day"] = df["Time"] % 86400
    df["Time_Hour_Sin"]    = np.sin(2 * np.pi * df["Time_Seconds_Day"] / 86400)
    df["Time_Hour_Cos"]    = np.cos(2 * np.pi * df["Time_Seconds_Day"] / 86400)
    df["Time_Hours"]       = df["Time"] / 3600
    df["Class_Label"]      = df["Class"].map({0: "Legítima", 1: "Fraude"})
    return df

@st.cache_resource
def carregar_modelo():
    try:
        modelo    = joblib.load(BASE / "models" / "modelo_final.pkl")
        scaler    = joblib.load(BASE / "models" / "scaler.pkl")
        threshold = joblib.load(BASE / "models" / "threshold_final.pkl")
        features  = joblib.load(BASE / "models" / "features.pkl")
        return modelo, scaler, threshold, features
    except Exception:
        return None, None, 0.5, None

df = carregar_dados()
modelo, scaler, threshold, FEATURES = carregar_modelo()

# Split temporal (igual ao notebook 02)
df_sorted = df.sort_values("Time").reset_index(drop=True)
cutoff    = df_sorted["Time"].quantile(0.80)
df_test   = df_sorted[df_sorted["Time"] > cutoff].copy()

if modelo and FEATURES:
    X_test_sc = pd.DataFrame(
        scaler.transform(df_test[FEATURES]), columns=FEATURES
    )
    df_test["score_fraude"] = modelo.predict_proba(X_test_sc)[:, 1]
    df_test["predito"]      = (df_test["score_fraude"] >= threshold).astype(int)
    df_test["pred_label"]   = df_test["predito"].map({0: "Legítima", 1: "Fraude"})
    y_test = df_test["Class"]
    y_pred = df_test["predito"]
    y_prob = df_test["score_fraude"]
else:
    # Fallback sem modelo — usar scores simulados para demonstração
    np.random.seed(42)
    scores = np.where(
        df_test["Class"] == 1,
        np.random.beta(8, 2, len(df_test)),
        np.random.beta(1, 8, len(df_test)),
    )
    df_test["score_fraude"] = scores
    df_test["predito"]      = (scores >= threshold).astype(int)
    df_test["pred_label"]   = df_test["predito"].map({0: "Legítima", 1: "Fraude"})
    y_test = df_test["Class"]
    y_pred = df_test["predito"]
    y_prob = df_test["score_fraude"]

# Métricas calculadas uma vez
cm                       = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp           = cm.ravel()
recall_val               = recall_score(y_test, y_pred)
precision_val            = precision_score(y_test, y_pred, zero_division=0)
f1_val                   = f1_score(y_test, y_pred)
f2_val                   = fbeta_score(y_test, y_pred, beta=2)
auc_val                  = roc_auc_score(y_test, y_prob)
pr_auc_val               = average_precision_score(y_test, y_prob)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Fraud Detection")
    st.markdown("**Sistema de Monitoramento de Risco**")
    st.markdown("---")

    st.markdown("### ⚙️ Threshold")
    thr_slider = st.slider(
        "Sensibilidade do modelo",
        min_value=0.01, max_value=0.99,
        value=float(threshold), step=0.01,
        help="Menor threshold = mais sensível (mais recall, mais falsos positivos)"
    )
    y_pred_thr = (y_prob >= thr_slider).astype(int)
    rec_thr    = recall_score(y_test, y_pred_thr)
    prec_thr   = precision_score(y_test, y_pred_thr, zero_division=0)
    fn_thr     = int(confusion_matrix(y_test, y_pred_thr).ravel()[2])
    fp_thr     = int(confusion_matrix(y_test, y_pred_thr).ravel()[1])

    st.markdown(f"""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:6px;padding:12px;font-family:'IBM Plex Mono',monospace;font-size:0.80rem">
    <div style="color:#8B949E">Threshold atual</div>
    <div style="color:#58A6FF;font-size:1.2rem;font-weight:600">{thr_slider:.3f}</div>
    <div style="color:#3FB950;margin-top:6px">Recall: {rec_thr:.1%}</div>
    <div style="color:#D29922">Precisão: {prec_thr:.1%}</div>
    <div style="color:#F85149">FN: {fn_thr} fraudes perdidas</div>
    <div style="color:#8B949E">FP: {fp_thr:,} bloqueadas</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Modelo")
    st.markdown(
        f"**Threshold ótimo:** `{threshold:.4f}`  \n"
        f"**PR-AUC:** `{pr_auc_val:.4f}`  \n"
        f"**AUC-ROC:** `{auc_val:.4f}`"
    )
    st.markdown("---")
    st.caption("Portfólio · Ciência de Dados")

# ─────────────────────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────────────────────
aba1, aba2, aba3, aba4 = st.tabs([
    "📡  Monitoramento",
    "🔬  Análise de Fraudes",
    "📈  Desempenho do Modelo",
    "🔎  Analisar Transação",
])

# ══════════════════════════════════════════════════════════
# ABA 1 — MONITORAMENTO
# ══════════════════════════════════════════════════════════
with aba1:
    st.markdown('<h2 style="color:#E6EDF3;">📡 Monitoramento em Tempo Real</h2>', unsafe_allow_html=True)
    st.caption(f"Conjunto de teste: {len(df_test):,} transações | Threshold: {thr_slider:.3f}")

    # Alerta de status
    if fn_thr == 0:
        st.markdown('<div class="alert-box ok">✅ SISTEMA OPERACIONAL — Nenhuma fraude não detectada com o threshold atual</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box">⚠️ ALERTA — {fn_thr} FRAUDE(S) NÃO DETECTADA(S) COM O THRESHOLD ATUAL</div>',
                    unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    taxa_fraude = df_test["Class"].mean() * 100

    for col, cls, label, val, delta in [
        (c1, "danger",  "Taxa de Fraude",   f"{taxa_fraude:.3f}%",   f"{int(df_test['Class'].sum())} transações"),
        (c2, "danger",  "Fraudes Perdidas",  f"{fn_thr}",             "falsos negativos"),
        (c3, "warning", "FP (bloqueadas)",   f"{fp_thr:,}",           "legítimas bloqueadas"),
        (c4, "success", "Recall",            f"{rec_thr:.1%}",        "fraudes detectadas"),
        (c5, "",        "Precisão",          f"{prec_thr:.1%}",       "acertos do alerta"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Volume de Transações por Hora</div>', unsafe_allow_html=True)
        df_test["hora"] = (df_test["Time_Hours"] % 24).astype(int)
        vol = df_test.groupby(["hora", "Class_Label"]).size().reset_index(name="n")
        fig = px.bar(vol, x="hora", y="n", color="Class_Label",
                     color_discrete_map={"Legítima": COR_LEGIT, "Fraude": COR_FRAUDE},
                     barmode="stack",
                     labels={"hora": "Hora do Dia", "n": "Transações", "Class_Label": "Tipo"})
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Hora do Dia", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            yaxis=dict(title="Transações", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Taxa de Fraude por Hora do Dia</div>', unsafe_allow_html=True)
        taxa_hora = df_test.groupby("hora")["Class"].mean() * 100
        fig = px.line(x=taxa_hora.index, y=taxa_hora.values,
                      labels={"x": "Hora do Dia", "y": "Taxa de Fraude (%)"},
                      markers=True)
        fig.update_traces(line_color=COR_FRAUDE, line_width=2.5,
                          marker=dict(size=6, color=COR_FRAUDE))
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Hora do Dia", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            yaxis=dict(title="Taxa (%)", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Distribuição de Amount: Fraude vs Legítima</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for label, cor in [("Legítima", COR_LEGIT), ("Fraude", COR_FRAUDE)]:
            sub = df_test[df_test["Class_Label"] == label]["Amount"].clip(upper=500)
            fig.add_trace(go.Histogram(x=sub, name=label, marker_color=cor,
                                       opacity=0.7, nbinsx=50, marker_line_width=0))
        fig.update_layout(**TEMPLATE, barmode="overlay",
            xaxis=dict(title="Amount (truncado em $500)", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            yaxis=dict(title="Frequência", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Scores de Risco — Distribuição</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for label, cor in [("Legítima", COR_LEGIT), ("Fraude", COR_FRAUDE)]:
            sub = df_test[df_test["Class_Label"] == label]["score_fraude"]
            fig.add_trace(go.Histogram(x=sub, name=label, marker_color=cor,
                                       opacity=0.7, nbinsx=50, marker_line_width=0))
        fig.add_vline(x=thr_slider, line_dash="dash", line_color="#58A6FF",
                      line_width=2, annotation_text=f"Threshold ({thr_slider:.3f})",
                      annotation_font_color="#58A6FF")
        fig.update_layout(**TEMPLATE, barmode="overlay",
            xaxis=dict(title="Score de Risco", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            yaxis=dict(title="Frequência", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ABA 2 — ANÁLISE DE FRAUDES
# ══════════════════════════════════════════════════════════
with aba2:
    st.markdown('<h2 style="color:#E6EDF3;">🔬 Análise de Fraudes Detectadas</h2>', unsafe_allow_html=True)

    fraudes_detectadas = df_test[(df_test["Class"] == 1) & (df_test["predito"] == 1)]
    fraudes_perdidas   = df_test[(df_test["Class"] == 1) & (df_test["predito"] == 0)]
    falsos_positivos   = df_test[(df_test["Class"] == 0) & (df_test["predito"] == 1)]

    c1, c2, c3 = st.columns(3)
    for col, cls, label, val, sub in [
        (c1, "success", "Fraudes Detectadas (TP)", f"{tp}", f"R$ {fraudes_detectadas['Amount'].sum():,.2f} protegidos"),
        (c2, "danger",  "Fraudes Perdidas (FN)",   f"{fn}", f"R$ {fraudes_perdidas['Amount'].sum():,.2f} em risco"),
        (c3, "warning", "Falsos Positivos (FP)",   f"{fp:,}", f"clientes legítimos bloqueados"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Comparação: Amount das Fraudes Detectadas vs Perdidas</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        if len(fraudes_detectadas) > 0:
            fig.add_trace(go.Box(y=fraudes_detectadas["Amount"], name="Detectadas (TP)",
                                  marker_color=COR_LEGIT, boxmean=True))
        if len(fraudes_perdidas) > 0:
            fig.add_trace(go.Box(y=fraudes_perdidas["Amount"], name="Perdidas (FN)",
                                  marker_color=COR_FRAUDE, boxmean=True))
        fig.update_layout(**TEMPLATE,
            yaxis=dict(title="Amount (R$)", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            xaxis=dict(title="", tickfont=dict(color="#C9D1D9")),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Score de Risco: Fraudes Detectadas vs Perdidas</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        if len(fraudes_detectadas) > 0:
            fig.add_trace(go.Histogram(x=fraudes_detectadas["score_fraude"],
                                        name="Detectadas (TP)", marker_color=COR_LEGIT,
                                        opacity=0.7, nbinsx=30))
        if len(fraudes_perdidas) > 0:
            fig.add_trace(go.Histogram(x=fraudes_perdidas["score_fraude"],
                                        name="Perdidas (FN)", marker_color=COR_FRAUDE,
                                        opacity=0.7, nbinsx=30))
        fig.add_vline(x=thr_slider, line_dash="dash", line_color="#58A6FF",
                      line_width=2)
        fig.update_layout(**TEMPLATE, barmode="overlay",
            xaxis=dict(title="Score de Risco", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            yaxis=dict(title="Frequência", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D"),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tabela de transações de alto risco
    st.markdown('<div class="section-title">Transações de Alto Risco (Score > 0.8)</div>',
                unsafe_allow_html=True)
    alto_risco = df_test[df_test["score_fraude"] > 0.8][
        ["Time_Hours", "Amount", "score_fraude", "Class_Label", "pred_label"]
    ].rename(columns={
        "Time_Hours": "Hora", "Amount": "Valor (R$)",
        "score_fraude": "Score", "Class_Label": "Real", "pred_label": "Predito"
    }).sort_values("Score", ascending=False).head(20)
    alto_risco["Score"] = alto_risco["Score"].round(4)
    alto_risco["Valor (R$)"] = alto_risco["Valor (R$)"].round(2)
    alto_risco["Hora"] = alto_risco["Hora"].round(1)
    st.dataframe(alto_risco, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# ABA 3 — DESEMPENHO DO MODELO
# ══════════════════════════════════════════════════════════
with aba3:
    st.markdown('<h2 style="color:#E6EDF3;">📈 Desempenho do Modelo</h2>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, cls, label, val, desc in [
        (c1, "success", "PR-AUC",   f"{pr_auc_val:.4f}", "Métrica principal"),
        (c2, "success", "AUC-ROC",  f"{auc_val:.4f}",    "Separação de classes"),
        (c3, "success", "Recall",   f"{recall_val:.1%}",  "Fraudes detectadas"),
        (c4, "warning", "Precisão", f"{precision_val:.1%}", "Acertos do alerta"),
        (c5, "",        "F2-Score", f"{f2_val:.4f}",      "Recall com peso 2×"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.5rem">{val}</div>
                <div class="kpi-delta">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Curva Precision-Recall</div>', unsafe_allow_html=True)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        baseline_pr = y_test.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines",
                                  name=f"Modelo (PR-AUC={pr_auc_val:.3f})",
                                  line=dict(color=COR_ACCENT, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[baseline_pr, baseline_pr],
                                  mode="lines", name=f"Aleatório ({baseline_pr:.4f})",
                                  line=dict(color="#8B949E", dash="dash", width=1.5)))
        # Marcar threshold atual
        prec_thr_c, rec_thr_c = precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred)
        fig.add_trace(go.Scatter(x=[rec_thr_c], y=[prec_thr_c], mode="markers",
                                  name=f"Threshold ({threshold:.3f})",
                                  marker=dict(color=COR_FRAUDE, size=12, symbol="star")))
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Recall", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D", range=[0,1]),
            yaxis=dict(title="Precisão", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D", range=[0,1.05]),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Curva ROC</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"Modelo (AUC={auc_val:.3f})",
                                  line=dict(color=COR_ACCENT, width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  name="Aleatório (AUC=0.500)",
                                  line=dict(color="#8B949E", dash="dash", width=1.5)))
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Taxa de Falso Positivo", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D", range=[-0.01,1.01]),
            yaxis=dict(title="Recall (TPR)", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9"), gridcolor="#21262D", range=[-0.01,1.01]),
            legend=dict(font=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Matriz de confusão
    st.markdown('<div class="section-title">Matriz de Confusão</div>', unsafe_allow_html=True)
    col_cm, col_exp = st.columns([1, 2])

    with col_cm:
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predito", y="Real", color="Count"),
            x=["Legítima", "Fraude"], y=["Legítima", "Fraude"],
            color_continuous_scale=[[0, "#0D1117"], [1, "#58A6FF"]],
            aspect="equal"
        )
        fig.update_traces(textfont=dict(color="white", size=16, family="IBM Plex Mono"))
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Predito", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9")),
            yaxis=dict(title="Real", title_font=dict(color="#C9D1D9"),
                       tickfont=dict(color="#C9D1D9")),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_exp:
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;padding:20px;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;line-height:2">
        <div style="color:#8B949E;font-size:0.70rem;letter-spacing:0.10em;text-transform:uppercase">Interpretação</div>
        <br>
        <div style="color:#3FB950">✅ TP = {tp:,} — Fraudes corretamente detectadas</div>
        <div style="color:#F85149">❌ FN = {fn} — Fraudes que escaparam (custo alto)</div>
        <div style="color:#D29922">⚠️ FP = {fp:,} — Legítimas bloqueadas (custo médio)</div>
        <div style="color:#58A6FF">🔵 TN = {tn:,} — Legítimas corretamente liberadas</div>
        <br>
        <div style="color:#8B949E;font-size:0.75rem">
        Prioridade: minimizar FN.<br>
        Cada FN = fraude não bloqueada = prejuízo direto.<br>
        FP = cliente legítimo frustrado, mas recuperável.
        </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ABA 4 — ANALISAR TRANSAÇÃO
# ══════════════════════════════════════════════════════════
with aba4:
    st.markdown('<h2 style="color:#E6EDF3;">🔎 Analisar Transação Individual</h2>', unsafe_allow_html=True)
    st.markdown("Informe os dados da transação para calcular o score de risco em tempo real.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💰 Dados da Transação**")
        amount_p = st.number_input("Valor da transação (R$)", 0.0, 50000.0, 100.0, 10.0)
        time_p   = st.slider("Hora do dia (0–23h)", 0, 23, 12)

        st.markdown("**📊 Componentes de Maior Risco (V1–V5)**")
        st.caption("Valores típicos: entre -3 e 3. Valores extremos aumentam o risco.")
        v1 = st.slider("V1", -20.0, 10.0, 0.0, 0.1)
        v2 = st.slider("V2", -10.0, 20.0, 0.0, 0.1)
        v3 = st.slider("V3", -20.0, 10.0, 0.0, 0.1)
        v4 = st.slider("V4", -5.0,  20.0, 0.0, 0.1)
        v5 = st.slider("V5", -15.0, 10.0, 0.0, 0.1)

    with col2:
        st.markdown("**📊 Outros Componentes (V6–V10)**")
        v6  = st.slider("V6",  -10.0, 10.0, 0.0, 0.1)
        v7  = st.slider("V7",  -40.0, 20.0, 0.0, 0.1)
        v8  = st.slider("V8",  -10.0, 20.0, 0.0, 0.1)
        v9  = st.slider("V9",  -10.0, 10.0, 0.0, 0.1)
        v10 = st.slider("V10", -20.0, 10.0, 0.0, 0.1)
        st.markdown("**📊 Componentes V11–V14**")
        v11 = st.slider("V11", -5.0,  10.0, 0.0, 0.1)
        v12 = st.slider("V12", -15.0,  5.0, 0.0, 0.1)
        v13 = st.slider("V13", -5.0,  10.0, 0.0, 0.1)
        v14 = st.slider("V14", -20.0,  5.0, 0.0, 0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Calcular Score de Risco", use_container_width=True, type="primary"):
        if modelo and FEATURES:
            # Montar vetor de entrada (demais V zerados)
            entrada_dict = {f: 0.0 for f in FEATURES}
            entrada_dict.update({
                "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
                "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
                "V11": v11, "V12": v12, "V13": v13, "V14": v14,
                "Amount_Log":    np.log1p(amount_p),
                "Time_Hour_Sin": np.sin(2 * np.pi * (time_p * 3600) / 86400),
                "Time_Hour_Cos": np.cos(2 * np.pi * (time_p * 3600) / 86400),
            })
            entrada = pd.DataFrame([entrada_dict])[FEATURES]
            entrada_sc = scaler.transform(entrada)
            score  = modelo.predict_proba(entrada_sc)[0, 1]
            nivel  = "FRAUDE" if score >= thr_slider else "LEGÍTIMA"
            cor    = COR_FRAUDE if nivel == "FRAUDE" else COR_LEGIT
            cls    = "danger" if nivel == "FRAUDE" else "success"

            st.markdown("---")
            r1, r2 = st.columns(2)

            with r1:
                st.markdown(f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-label">Score de Risco</div>
                    <div class="kpi-value" style="color:{cor};font-size:2.5rem">{score:.1%}</div>
                    <div class="kpi-delta">probabilidade de fraude</div>
                </div>
                <div class="kpi-card {cls}" style="margin-top:8px">
                    <div class="kpi-label">Classificação</div>
                    <div class="kpi-value" style="color:{cor};font-size:1.8rem">{nivel}</div>
                    <div class="kpi-delta">threshold: {thr_slider:.3f}</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={"suffix": "%", "font": {"size": 32, "color": cor,
                                                    "family": "IBM Plex Mono"}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%",
                                 "tickfont": {"color": "#C9D1D9"}},
                        "bar": {"color": cor, "thickness": 0.3},
                        "bgcolor": "#161B22",
                        "bordercolor": "#30363D",
                        "steps": [
                            {"range": [0,    thr_slider*50],  "color": "#0D2B1A"},
                            {"range": [thr_slider*50, thr_slider*100], "color": "#2D2000"},
                            {"range": [thr_slider*100, 100],  "color": "#2D0000"},
                        ],
                        "threshold": {
                            "line": {"color": "#58A6FF", "width": 3},
                            "thickness": 0.8,
                            "value": thr_slider * 100,
                        },
                    },
                ))
                fig.update_layout(
                    height=260, margin=dict(t=20, b=10, l=30, r=30),
                    paper_bgcolor="#0D1117",
                    font=dict(family="IBM Plex Mono", color="#C9D1D9"),
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Modelo não encontrado. Execute `02_modelagem.ipynb` para gerar os arquivos em `models/`.")
