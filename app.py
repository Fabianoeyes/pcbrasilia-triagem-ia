import unicodedata

def norm(txt: str) -> str:
    """Remove acentos, deixa minusculo e normaliza espaços."""
    if not txt:
        return ""
    txt = txt.strip().lower()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join([c for c in txt if not unicodedata.combining(c)])
    return " ".join(txt.split())

# Base (didática) - você pode crescer isso à vontade
KB = [
    {
        "tags": ["preservacao de local", "isolamento", "vestigios", "local de crime"],
        "texto": (
            "Em ocorrências graves, priorize a preservação do local: "
            "evite contaminação de vestígios, controle acesso e registre informações essenciais "
            "conforme protocolos institucionais (base didática)."
        ),
    },
    {
        "tags": ["violencia domestica", "vd", "maria da penha", "agressao domestica"],
        "texto": (
            "Violência doméstica (base didática): priorize a segurança da vítima; avalie risco imediato; "
            "oriente a preservação de evidências (mensagens, áudios, fotos, laudos); "
            "registre circunstâncias com clareza e oriente sobre medidas protetivas e canais formais, "
            "conforme protocolos vigentes."
        ),
    },
    {
        "tags": ["homicidio", "tentativa de homicidio", "lesao grave", "risco a vida"],
        "texto": (
            "Homicídio / tentativa (base didática): trate como alta criticidade. "
            "Atenção ao acionamento de equipes competentes, preservação do local e registro de informações "
            "sobre vítimas, testemunhas e dinâmica inicial, conforme protocolos institucionais."
        ),
    },
    {
        "tags": ["estelionato", "golpe", "fraude", "pix", "cartao"],
        "texto": (
            "Estelionato (base didática): coletar evidências digitais (comprovantes, prints, links, contas, conversas); "
            "orientar preservação de registros e canais formais para bloqueio/contestação quando aplicável; "
            "registrar a narrativa com datas, valores e identificadores."
        ),
    },
    {
        "tags": ["ameaca", "intimidacao", "coacao"],
        "texto": (
            "Ameaça (base didática): registrar circunstâncias, identificar meio (presencial/mensagem), "
            "avaliar risco e orientar preservação de evidências (mensagens, áudios, prints)."
        ),
    },
]

def recuperar_rag(pergunta: str) -> str:
    q = norm(pergunta)

    # Pontuação simples: conta quantas tags bateram
    melhor = None
    melhor_score = 0

    for item in KB:
        score = 0
        for tag in item["tags"]:
            if norm(tag) in q:
                score += 1
        if score > melhor_score:
            melhor_score = score
            melhor = item["texto"]

    if melhor_score == 0:
        return "Não consta na base um procedimento específico para esse tema (base didática)."

    return melhor

