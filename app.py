import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import unicodedata

st.set_page_config(page_title="Triagem Inteligente - PC Brasília", layout="centered")

st.title("Triagem Inteligente – PC Brasília (Demo)")
st.caption("Modelo didático com Machine Learning + Mini-RAG")

# =========================
# 1) MODELO DE MACHINE LEARNING
# =========================

@st.cache_data
def treinar_modelo():
    data = [
        ["Violência Doméstica","Taguatinga","Madrugada","Sim","Sim","Sim","Alta"],
        ["Furto","Plano Piloto","Tarde","Não","Não","Não","Baixa"],
        ["Ameaça","Ceilândia","Noite","Sim","Não","Sim","Média"],
        ["Homicídio","Samambaia","Noite","Sim","Sim","Não","Alta"],
        ["Estelionato","Asa Norte","Manhã","Não","Não","Sim","Média"],
    ]
    cols = ["tipo","local","periodo","tem_arma","vitima_ferida","reincidencia","prioridade"]
    df = pd.DataFrame(data, columns=cols)

    X = df.drop("prioridade", axis=1)
    y = df["prioridade"]

    cat_cols = X.columns
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe

pipe = treinar_modelo()

st.header("1) Classificação da Ocorrência")

tipo = st.selectbox("Tipo", ["Violência Doméstica","Furto","Ameaça","Homicídio","Estelionato"])
local = st.selectbox("Local", ["Plano Piloto","Taguatinga","Ceilândia","Samambaia","Asa Norte"])
periodo = st.selectbox("Período", ["Manhã","Tarde","Noite","Madrugada"])
tem_arma = st.selectbox("Tem arma?", ["Sim","Não"])
vitima_ferida = st.selectbox("Vítima ferida?", ["Sim","Não"])
reinc = st.selectbox("Reincidência?", ["Sim","Não"])

if st.button("Classificar prioridade"):
    nova = pd.DataFrame([{
        "tipo":tipo,"local":local,"periodo":periodo,
        "tem_arma":tem_arma,"vitima_ferida":vitima_ferida,"reincidencia":reinc
    }])
    p = pipe.predict(nova)[0]
    st.success(f"Prioridade prevista: **{p}**")

# =========================
# 2) MINI-RAG (BASE DE CONHECIMENTO)
# =========================

def norm(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")

KB = [
    {"tema":"violencia domestica","texto":"Priorizar segurança da vítima, aplicar Lei Maria da Penha, registrar ocorrência, avaliar medidas protetivas, preservar evidências e acionar rede de apoio."},
    {"tema":"homicidio","texto":"Isolar local do crime, preservar vestígios, acionar perícia, identificar testemunhas, registrar BO e comunicar autoridade policial competente."},
    {"tema":"ameaca","texto":"Registrar circunstâncias, identificar meio (presencial/mensagem), avaliar risco, orientar preservação de mensagens e áudios."},
    {"tema":"furto","texto":"Registrar ocorrência, coletar informações do bem, imagens de câmeras, orientar bloqueio de dispositivos e rastreamento."},
    {"tema":"estelionato","texto":"Coletar evidências digitais, orientar bloqueio bancário, registrar BO, preservar prints e logs."},
]

def recuperar_rag(pergunta):
    q = norm(pergunta)
    for item in KB:
        if item["tema"] in q:
            return item["texto"]
    return "Não consta procedimento específico na base didática."

st.header("2) Assistente (Mini-RAG)")

q = st.text_input("Pergunta (ex: Como tratar um caso de Violência Doméstica?)")
if q:
    st.info(recuperar_rag(q))
