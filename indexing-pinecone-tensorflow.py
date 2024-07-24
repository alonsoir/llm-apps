import os

import pinecone
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from collections import defaultdict
import psutil

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB")

def init_pinecone(api_key, environment, index_name):
    try:
        pinecone.init(api_key=api_key, environment=environment)
        index = pinecone.Index(index_name)
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None


def load_models(model_names):
    tokenizers = []
    models = []
    for name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = TFAutoModel.from_pretrained(name)
            tokenizers.append(tokenizer)
            models.append(model)
        except Exception as e:
            print.f("Error loading model {name}: {e}")
    return tokenizers, models


def generate_embeddings(text, tokenizers, models):
    embeddings = []
    for tokenizer, model in zip(tokenizers, models):
        try:
            inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
            outputs = model(inputs)
            embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embeddings for model: {e}")
    return embeddings


def query_pinecone(index, embeddings):
    results = []
    try:
        for embedding in embeddings:
            result = index.query(vector=embedding, top_k=10)
            results.append(result)
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
    return results


def combine_results(results):
    combined_results = defaultdict(float)
    try:
        for result in results:
            for match in result['matches']:
                combined_results[match['id']] += match['score']
    except Exception as e:
        print(f"Error combining results: {e}")
    final_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    return final_results


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_API_ENV")  # Añade tu host de Pinecone
    index_name = 'medical-index'
    model_names = ['dmis-lab/biobert-base-cased-v1.1', 'emilyalsentzer/Bio_ClinicalBERT', 'custom-model-rare-diseases']

    # Inicializar Pinecone
    index = init_pinecone(api_key, environment, index_name)
    if not index:
        return

    # Cargar modelos
    tokenizers, models = load_models(model_names)
    if not tokenizers or not models:
        return

    # Consulta de ejemplo
    query = "tratamientos para la enfermedad rara X"

    # Generar embeddings para la consulta utilizando múltiples modelos
    query_embeddings = generate_embeddings(query, tokenizers, models)
    if not query_embeddings:
        return

    # Consultar Pinecone con los embeddings
    results = query_pinecone(index, query_embeddings)
    if not results:
        return

    # Combinar los resultados de múltiples consultas
    final_results = combine_results(results)

    # Mostrar resultados finales
    for result in final_results:
        print(f"Document ID: {result[0]}, Score: {result[1]}")

    print_memory_usage()


if __name__ == "__main__":
    main()
