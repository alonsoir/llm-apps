import os

import pinecone
from transformers import AutoTokenizer, AutoModel
import torch

def init_pinecone(api_key, environment, index_name):
    try:
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=768)  # Asume que los embeddings son de dimensión 768
        index = pinecone.Index(index_name)
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def generate_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embeddings for text: {e}")
    return embeddings

def index_documents(index, texts, embeddings):
    try:
        vectors = [{'id': str(i), 'values': embedding.tolist()} for i, embedding in enumerate(embeddings)]
        index.upsert(vectors)
    except Exception as e:
        print(f"Error indexing documents: {e}")

def main_indexing():
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_API_ENV")  # Añade tu host de Pinecone
    index_name = 'medical-index'
    model_name = 'dmis-lab/biobert-base-cased-v1.1'

    # Inicializar Pinecone
    index = init_pinecone(api_key, environment, index_name)
    if not index:
        return

    # Cargar modelo
    tokenizer, model = load_model(model_name)
    if not tokenizer or not model:
        return

    # Documentos de ejemplo a indexar
    documents = [
        "Document 1 text...",
        "Document 2 text...",
        "Document 3 text..."
    ]

    # Generar embeddings
    embeddings = generate_embeddings(documents, tokenizer, model)
    if not embeddings:
        return

    # Indexar documentos en Pinecone
    index_documents(index, documents, embeddings)

if __name__ == "__main__":
    main_indexing()
