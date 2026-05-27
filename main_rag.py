from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=api_key
)
"""
Alternativa gratuita completamente:
1.Baixar : pip install sentence-transformers langchain-huggingface
2.Usar : 
from langchain_huggingface import HuggingFaceEmbeddings

# Substituindo a linha: embeddings = GoogleGenerativeAIEmbeddings(...) por:
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
"""

# documento = TextLoader(
#     "documentos/GTB_gold_Nov23.txt",
#     encoding="utf-8"
# ).load()

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf"
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
)

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
).split_documents(documentos)

dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k": 2})

prompt_consuta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consuta_seguro | modelo | StrOutputParser()


def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke(
        {"query": pergunta, "contexto": contexto
         })


print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"))
