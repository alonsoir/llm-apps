from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Asegúrate de que PyTorch esté utilizando el dispositivo correcto (CPU en este caso)
device = torch.device('cpu')

# Cargar el PDF
pdf_path = "formulas.pdf"
pdf = PdfReader(pdf_path)

# Extraer el texto del PDF
text = ""
for page in pdf.pages:
    text += page.extract_text()

# Dividir el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)

# Crear documentos
documents = [Document(page_content=chunk) for chunk in chunks]

# Usar HuggingFaceEmbeddings en lugar de una clase personalizada
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Crear un almacén de vectores local usando FAISS
vector_store = FAISS.from_documents(documents, embedding=embeddings)

# Configurar el LLM (en este caso, usando OpenAI)
openai_api_key = os.environ.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Crear la cadena de recuperación de preguntas y respuestas
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Hacer una pregunta sobre el PDF
question = "¿Cuál es el tema principal del PDF?"
answer = qa_chain.invoke(question)
print(f"Pregunta: {question}")
print(f"Respuesta: {answer['result']}")