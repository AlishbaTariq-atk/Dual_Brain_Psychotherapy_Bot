from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
# Load environment variables
load_dotenv()

df = pd.read_csv("transcripts.csv")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2,
    chunk_overlap=1,
    separators="\n\n"
)

chunks=splitter.create_documents(df["Structured_Transcript"].tolist())

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store= FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 10})
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = retriever.invoke('Fredric Schiffer?')
context= " ".join([doc.page_content for doc in result])
response = model.invoke([HumanMessage(content=f"Context: {context}\n\nQuestion: how can i cure my anxiety?")])
print(response.content)
