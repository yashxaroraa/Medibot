from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document  # type: ignore # Import Document class
import pdfplumber  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore

# Step 1: Load a Single PDF
DATA_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

def load_pdf_with_pdfplumber(file_path):
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # Only add non-empty pages
                pages.append(Document(page_content=text))  # ✅ Wrap in Document object
    return pages

documents = load_pdf_with_pdfplumber(DATA_PATH)
# print("Length of PDF pages:", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)  # ✅ Now passing `Document` objects
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
# print("Length of text chunks:", len(text_chunks))


# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# step 4: store embeddings in FAISS 
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
