from pinecone import Pinecone, ServerlessSpec, PodSpec
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import defaultdict
import os

def init_pinecone(api_key, environment, index_name):
    try:
        pc = Pinecone(
            api_key=api_key
        )
        # Now do stuff
        if index_name not in pc.list_indexes().names():
            return pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment),
                deletion_protection="enabled"
            )
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None


def load_models(model_names):
    tokenizers = []
    models = []
    for name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            tokenizers.append(tokenizer)
            models.append(model)
        except Exception as e:
            print(f"Error loading model {name}: {e}")
    return tokenizers, models


def generate_embeddings(text, tokenizers, models):
    embeddings = []
    for tokenizer, model in zip(tokenizers, models):
        try:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding.detach().numpy())
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
    index_name = 'test-medical-index'
    model_names = ['dmis-lab/biobert-base-cased-v1.1', 'emilyalsentzer/Bio_ClinicalBERT', 'custom-model-rare-diseases']

    # Initialize Pinecone
    index = init_pinecone(api_key, environment, index_name)
    if not index:
        return

    # Load models
    tokenizers, models = load_models(model_names)
    if not tokenizers or not models:
        return

    # Example query
    query = "tratamientos para la enfermedad rara X"

    # Generate embeddings for the query using multiple models
    query_embeddings = generate_embeddings(query, tokenizers, models)
    if not query_embeddings:
        return

    # Query Pinecone with the embeddings
    results = query_pinecone(index, query_embeddings)
    if not results:
        return

    # Combine the results from multiple queries
    final_results = combine_results(results)

    # Display final results
    for result in final_results:
        print(f"Document ID: {result[0]}, Score: {result[1]}")


if __name__ == "__main__":
    main()
