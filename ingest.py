from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # used to load and parse documents
from langchain.text_splitter import RecursiveCharacterTextSplitter # create chunks along with text overlapping
from langchain_huggingface import HuggingFaceEmbeddings #HuggingFaceEmbeddings class is used to generate embeddings
from langchain_community.vectorstores import FAISS #FAISS is used to store embeddings

DATA_PATH = 'data/' # stores the pdf from the data folder
DB_FAISS_PATH = 'vectorstore/db_faiss' # embeddings are stored inside vector store folder which 
                                        # is generated after ingest.py runs

#function to create vector_db
def create_vector_db():

    #DirectoryLoader loads all files from a folder.
    #PyPDF loads a single PDF file.
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf',loader_cls=PyPDFLoader)

    documents = loader.load()

    #overlapping chunks are created to preserve the context between the chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device':'cpu'}) # embeddings are created
    db=FAISS.from_documents(texts, embeddings) # creates a FAISS vector store
    db.save_local(DB_FAISS_PATH) # saves the folder locally. Path is stored inside DB_FAISS_PATH

if __name__ == "__main__":
    create_vector_db()
