from openai import OpenAI
from pymilvus import MilvusClient

MODEL_NAME = "text-embedding-3-small"  # Which model to use, please check https://platform.openai.com/docs/guides/embeddings for available models
DIMENSION = 1536  # Dimension of vector embedding

openai_client = OpenAI(api_key="YOUR_API_KEY")

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = [
    vec.embedding
    for vec in openai_client.embeddings.create(input=docs, model=MODEL_NAME).data
]

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(docs))
]


# milvus_client = MilvusClient(uri="milvus_openai_demo.db")
milvus_client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

COLLECTION_NAME = "demo_collection"  # Milvus collection name
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)
milvus_client.create_collection(collection_name=COLLECTION_NAME, dimension=DIMENSION)

res = milvus_client.insert(collection_name="demo_collection", data=data)

print(f"{res}\n")

queries = ["When was artificial intelligence founded?"]

query_vectors = [
    vec.embedding
    for vec in openai_client.embeddings.create(input=queries, model=MODEL_NAME).data
]

res = milvus_client.search(
    collection_name=COLLECTION_NAME,  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

for q in queries:
    print("Query:", q)
    for result in res:
        print(result)
    print("\n")

