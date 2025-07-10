from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader #used to load and parse pdf
from langchain.prompts import PromptTemplate #creates custom prompt template
from langchain_community.embeddings import HuggingFaceEmbeddings # used to create embeddings.
from langchain_community.vectorstores import FAISS # used to store embeddings
from langchain_community.llms import CTransformers # used to load llm
from langchain.chains import RetrievalQA # used to build QnA system over custom data
import chainlit as cl # used to create chatgpt like UI.
from langchain_community.llms import Ollama

DB_FAISS_PATH = 'vectorstore/db_faiss'

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

#Retrieval QA chain
def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',# injects all context in one prompt
                                           retriever=db.as_retriever(search_kwargs={'k':2}),#returns top 2 chunks
                                           return_source_documents = True,# returns text chunks used for answering the question
                                           chain_type_kwargs={'prompt':prompt}# injects custom prompt template
    )
    return qa_chain

#Loading the Model
def load_llm():
    llm=Ollama(
        model="llama3.2:1b",
        temperature=0.5,#determines creativity
        # max_tokens=512 # generates upto 512 tokens in response
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'}) # creates the question embeddings
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) # loads previously saved FAISS for searching
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_qa_chain(llm,qa_prompt,db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = await chain.acall(message.content)
    answer = res["result"]
    sources = res["source_documents"]
    source_texts = "\n\n".join(
        f"Source (page {doc.metadata.get('page_label', doc.metadata.get('page', ''))}): {doc.metadata.get('source', '')}"
        for doc in sources
    )
    msg = cl.Message(content=f"{answer}\n\n{source_texts}")
    await msg.send()

