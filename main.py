from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"]
)

"""
# Este modelo que deveria ser utilizado mas a API é paga.
modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)
"""

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    api_key=api_key
)

cadeia = prompt_cidade | modelo | StrOutputParser()

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)
print(resposta)
