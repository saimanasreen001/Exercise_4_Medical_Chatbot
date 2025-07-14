## LangChain --> Python framework that helps build apllications using LLMs.

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader #used to load and parse pdf
from langchain.prompts import PromptTemplate #creates custom prompt template
from langchain_community.embeddings import HuggingFaceEmbeddings # used to create embeddings.
from langchain_community.vectorstores import FAISS # used to store embeddings
from langchain.chains import RetrievalQA # used to build QnA system over custom data
import chainlit as cl                    # used to create chatgpt like UI.
from langchain_community.llms import Ollama # connects Langchain with Ollama LLM.

DB_FAISS_PATH = 'vectorstore/db_faiss' # path where embeddings are stored.

#custom prompt template tells llm what to do. The more better custom prompt template,better it is.
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question:{question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variable=['context','question'])
    return prompt

#Loading the Model
def load_llm():
    llm=Ollama(
        model="llama3.2:1b",
        temperature=0.5,#determines creativity
    )
    return llm

#Retrieval QA chain
def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',                              # injects all retrieved chunks/context into the prompt
                                           retriever=db.as_retriever(search_kwargs={'k':2}),# returns top 2 chunks from FAISS
                                           return_source_documents = True,                  # returns sources of the answer.
                                           chain_type_kwargs={'prompt':prompt}              # injects custom prompt template
    )
    return qa_chain

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})                     # loads embedding model
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) # loads previously saved FAISS for searching
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_qa_chain(llm,qa_prompt,db)

    return qa

#output function
def final_result(query): # takes user question
    qa_result = qa_bot() # returns retrieval qa chain/
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...") # first displays starting the bot
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?" # then displays this message
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):# message is the user's message
    chain = cl.user_session.get("chain")
    res = await chain.acall(message.content)
    answer = res["result"] # holds answer given by the chatbot
    sources = res["source_documents"] # holds answer resource.
    source_texts = "\n\n".join( # holds structure of source like pagelabel , page 
        f"Source (page {doc.metadata.get('page_label', doc.metadata.get('page', ''))}): {doc.metadata.get('source', '')}"
        for doc in sources
    )
    msg = cl.Message(content=f"{answer}\n\n{source_texts}") # displays answer and source together.
    await msg.send() # sends message on UI.

