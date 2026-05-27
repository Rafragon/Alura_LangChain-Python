from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# estava dando errado uma parte do codigo e por isso decidi usar o pydantic
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    api_key=api_key
)

prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens"),
        ("human", "{query}")
    ]
)

assistente = prompt_consultor | modelo | StrOutputParser()

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra. Praia. Você é uma especialista em viagens com destinos para praias."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr. Montanha. Você é uma especialista em viagens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()


class Rota(BaseModel):  # era TypedDict
    destino: Literal["praia", "montanha"]


# ALTERAÇÃO: O prompt agora orienta a classificação sem quebrar o esquema JSON do TypedDict
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Classifique a solicitação do usuário identificando se o destino preferido é praia ou montanha."),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)


class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str


async def no_roteador(estado: Estado, config=RunnableConfig):
    resultado = await roteador.ainvoke({"query": estado["query"]}, config)
    return {"destino": resultado}


async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}


async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}


def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    rota_obj = estado.get("destino")

    if rota_obj and hasattr(rota_obj, "destino"):
        destino_final = str(rota_obj.destino).strip().lower()
        if "praia" in destino_final:
            return "praia"
    return "montanha"


grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()


async def main():
    resposta = await app.ainvoke({"query": "Quero visitar um lugar para aproveitar a vista e com ótima vista."})
    print(resposta["resposta"])


asyncio.run(main())
