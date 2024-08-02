from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoTokenizer, AutoModel
import openai
from databricks.sdk import WorkspaceClient

w = WorkspaceClient(debug_headers=True)
# ...


# Configuración del cliente de Vector Search de Databricks
vs_client = VectorSearchClient()

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

# Generar las incrustaciones utilizando un modelo de HuggingFace
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

embeddings = [embed_text(chunk) for chunk in chunks]

# Crear una tabla Delta con las incrustaciones
table_name = "pdf_embeddings"
vs_client.create_delta_table(
    table_name=table_name,
    path="/path/to/delta/table",
    embedding_dimension=embeddings[0].shape[1],
    primary_key="id"
)

# Insertar datos en la tabla Delta
data = [
    {"id": i, "text": chunk, "embedding": embedding.tolist()}
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]
vs_client.upsert(table_name, data)

# Crear el índice de búsqueda vectorial
index_name = "pdf_search_index"
vs_client.create_index(
    index_name=index_name,
    source_table_name=table_name,
    primary_key="id",
    embedding_dimension=embeddings[0].shape[1],
    embedding_column="embedding"
)

# Configurar LangChain para usar el índice de búsqueda vectorial
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Configurar el almacén de vectores de Databricks
embeddings = HuggingFaceEmbeddings()
vector_store = DatabricksVectorSearch(
    vs_client,
    index_name,
    text_column="text",
    embedding=embeddings
)

# Configurar el LLM (en este caso, usando OpenAI)
openai.api_key = "tu_api_key_de_openai"
llm = OpenAI(temperature=0)

# Crear la cadena de recuperación de preguntas y respuestas
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Hacer una pregunta sobre el PDF
question = "¿Cuál es el tema principal del PDF?"
answer = qa_chain.run(question)
print(f"Pregunta: {question}")
print(f"Respuesta: {answer}")