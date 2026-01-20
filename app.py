import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Triagem PC Brasília (Demo IA)", layout="centered")

st.title("Triagem Inteligente – PC Brasília (Demo)")
st.caption("Modelo didático com dados sintéticos (aula).")

# ========= 1) Dataset sintético + treino do modelo =========
@st.cache_resource
def treinar_modelo():
    np.random.seed(42)

    tipos = ["Furto", "Roubo", "Ameaça", "Violência Doméstica", "Estelionato", "Tráfico", "Homicídio (tentativa)"]
    locais = ["Asa Norte", "Asa Sul", "Ceilândia", "Taguatinga", "Samambaia", "Planaltina", "Sobradinho"]
    periodos = ["Madrugada", "Manhã", "Tarde", "Noite"]

    def definir_prioridade(row):
        score = 0
        if row["tipo"] in ["Homicídio (tentativa)", "Tráfico", "Violência Doméstica", "Roubo"]:
            score += 2
        if row["tem_arma"] == 1:
            score += 2
        if row["vitima_ferida"] == 1:
            score += 2
        if row["historico_reincidencia"] == 1:
            score += 1
        if score >= 5:
            return "Alta"
        elif score >= 3:
            return "Média"
        else:
            return "Baixa"

    n = 220
    df = pd.DataFrame({
        "tipo": np.random.choice(tipos, n),
        "local": np.random.choice(locais, n),
        "periodo": np.random.choice(periodos, n),
        "tem_arma": np.random.choice([0,1], n, p=[0.86, 0.14]),
        "vitima_ferida": np.random.choice([0,1], n, p=[0.76, 0.24]),
        "historico_reincidencia": np.random.choice([0,1], n, p=[0.72, 0.28])
    })
    df["prioridade"] = df.apply(definir_prioridade, axis=1)

    X = df[["tipo","local","periodo","tem_arma","vitima_ferida","historico_reincidencia"]]
    y = df["prioridade"]

    cat_cols = ["tipo","local","periodo"]
    num_cols = ["tem_arma","vitima_ferida","historico_reincidencia"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = RandomForestClassifier(n_estimators=250, random_state=42)

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    return pipe, tipos, locais, periodos

pipe, tipos, locais, periodos = treinar_modelo()

# ========= 2) Inputs do usuário =========
st.subheader("1) Informe os dados da ocorrência")

col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo", tipos)
    local = st.selectbox("Local", locais)
    periodo = st.selectbox("Período", periodos)

with col2:
    tem_arma = st.selectbox("Tem arma?", [0,1], format_func=lambda x: "Não" if x==0 else "Sim")
    vitima_ferida = st.selectbox("Vítima ferida?", [0,1], format_func=lambda x: "Não" if x==0 else "Sim")
    reinc = st.selectbox("Reincidência?", [0,1], format_func=lambda x: "Não" if x==0 else "Sim")

if st.button("Classificar prioridade"):
    nova = pd.DataFrame([{
        "tipo": tipo,
        "local": local,
        "periodo": periodo,
        "tem_arma": tem_arma,
        "vitima_ferida": vitima_ferida,
        "historico_reincidencia": reinc
    }])
    prioridade = pipe.predict(nova)[0]
    st.success(f"Prioridade prevista: **{prioridade}**")

# ========= 3) Mini-RAG =========
st.subheader("2) Assistente (Mini-RAG)")

kb = {
    "preservação de local": "Em ocorrências graves, orientar preservação do local, evitar contaminação de vestígios e acionar equipe competente conforme protocolos.",
    "violência doméstica": "Priorizar segurança da vítima, avaliar risco imediato, orientar registro e medidas protetivas conforme protocolos vigentes.",
    "estelionato": "Coletar evidências digitais (comprovantes, prints, contas), orientar preservação de registros e canais formais para bloqueio/contestação quando aplicável.",
    "ameaça": "Registrar circunstâncias, identificar meio (presencial/mensagem), avaliar risco e orientar preservação de evidências (mensagens, áudios)."
}

q = st.text_input("Pergunta (ex.: Como tratar um caso de Ameaça?)")
if q:
    ql = q.lower()
    contexto = None
    for tema, texto in kb.items():
        if tema in ql:
            contexto = texto
            break
    if not contexto:
        contexto = "Não consta na base um procedimento específico para esse tema (base didática)."
    st.info(contexto)
