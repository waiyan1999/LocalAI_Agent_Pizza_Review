from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os


df = pd.read_csv("realistic_restaurant_reviews.csv")

#print(df.head())

embedding = OllamaEmbeddings(
    model="mxbai-embed-large"
)


#print(embedding)

db_location = './chromadb_langchain.db'

add_document = not os.path.exists(db_location)


#print(add_document)

if add_document:
    ids = []
    documents = []
    
    for i,row in df.iterrows():
        
        document = Document(
            page_content=row["Title"]+" "+row["Review"],
            metadata = {
                "rating": row["Rating"],
                "date": row["Date"]
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        

vector_store = Chroma(
    collection_name="res_review",
    persist_directory=db_location,
    embedding_function=embedding
)

if add_document:
    vector_store.add_documents(documents=documents, ids= ids)
        
reteiver = vector_store.as_retriever()