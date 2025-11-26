from langchain_community.document_loaders import WebBaseLoader,SeleniumURLLoader
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

url='https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.986374/full'
loader=WebBaseLoader(url)

docs=loader.load()
soup=BeautifulSoup(docs[0].page_content, 'html.parser')
context=soup.get_text(strip=True)
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = PromptTemplate(
    template="""
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
parser = StrOutputParser()
chain = prompt | llm | parser
question = "What are the main findings of the study?"
result = chain.invoke({"context": context, "question": question})
print(result)
