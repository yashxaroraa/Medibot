import os
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document  # type: ignore # Import Document class
import pdfplumber  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA # type: ignore

from dotenv import load_dotenv  # type: ignore # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Now, get HF_TOKEN from the .env file
HF_TOKEN = os.getenv("HF_TOKEN")

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length": "512",}
    )
    return llm


# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context
Context: {context}
Question: {question}
Start the answer directly.
No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

DB_FAISS_PATH="vectorstore/db_faiss"
embeddingmodel=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embeddingmodel, allow_dangerous_deserialization=True)

# Create AQ chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever= db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents= True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query

user_query=input("Write query here:\n")
response=qa_chain.invoke({'query': user_query})
print("RESULT:\n", response["result"])
print("Source Documents:", response["source_documents"])

