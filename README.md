# Medical Chatbot

    This chatbot is able to answer user medical queries related to the document stored in the data folder. There are three different aspects ingestion, storage and retrieval. 

## Project Structure

        Exercise_4_Medical_Chatbot/
    ├── _pycache                
    ├── ingest.py               
    ├── model.py                   
    ├── requirements.txt           
    ├── README.md                  
    ├── data/                      
    │   └── 71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf
    ├── vectorstore/               
    │   └── db_faiss/
    │       ├── index.faiss
    │       └── index.pkl
    ├── .chainlit          
    ├── venv/          
    └── .gitignore 

## WorkFlow

    1. A document(.pdf) is stored inside the data folder. It is loaded and chunks are created out of which embeddings are created and stored in FAISS vector db. This talks about ingestion and storage.
    2. Embeddings are stored inside vectorstore folder locally.
    3. On the retrieval part, custom prompt template is defined by passing context and question as input parameters.
    4. Then the llm model llama3.2:1b is loaded using ollama.
    5. The vector db is fetched locally and qa chain is strated using llm model, prompt and the vector db. And final answer is received.
    6. UI is created using chainlit.

## Setup instructions

    1. Clone the repository

        git clone https://github.com/saimanasreen001/Exercise_4_Medical_Chatbot.git
        cd Exercise_4_Medical_Chatbot

    2. Create and activate the virtual environment

        python3 -m venv venv
        source venv/bin/activate

    3. Install the dependencies from requirements.txt

        pip install -r requirements.txt

    4. Pull the llama3.2:1b model using ollama.

    5. Create the vectorstore

        python ingest.py

    6. Run the chatbot

        chainlit run model.py


