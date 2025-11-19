from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pprint

from vector import reteiver

model = OllamaLLM(
    model="llama3"
)

template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template=template)

chain = prompt | model 

# pprint.pprint(chain)

while True:
    
    print('\n\n=========================')
    question = input("Ask to AI Agent : ")
    print('=============================')
    
    if question == 'q':
        print('\n Thanks for Asking AI Agent')
        break
        
    
    reviews = reteiver.invoke(question)
    
    result = chain.invoke(
        {
            'reviews':[],
            'question': question
        }
    )
    
    print(result)