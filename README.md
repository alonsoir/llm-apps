## 🧠 LLM App with Memory
This Streamlit app is an AI-powered chatbot that uses OpenAI's GPT-4o model with a persistent memory feature. 
It allows users to have conversations with the AI while maintaining context across multiple interactions.
Original source code uses qdrant for persistent vector db, i will include milvus too.

### Features

- Utilizes OpenAI's GPT-4o model for generating responses
- Implements persistent memory using Mem0 and Qdrant vector store
- Implements persistent memory using Mem0 and Milvus vector store. In progress.
- Allows users to view their conversation history
- Provides a user-friendly interface with Streamlit


### How to get Started?

1. Clone the GitHub repository
```bash
git clone https://github.com/alonsoir/llm-apps.git
```

2. Install the required dependencies:

```bash
  poetry shell
  poetry install
```

3. Ensure Qdrant is running:
The app expects Qdrant to be running on localhost:6333. Adjust the configuration in the code if your setup is different.

```bash
    docker pull qdrant/qdrant
    
    docker run -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
```
3.1 Ensure Milvus is running:
    The script test-milvus.py expects Milvus to be running on localhost:19530. Adjust the configuration in the code if your setup is
different. It will download etcd as an embedded database.
```
    mkdir milvus && cd milvus && 
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
    
    bash standalone_embed.sh start
    
    Password:
    Unable to find image 'milvusdb/milvus:v2.4.5' locally
    2024/07/23 13:17:27 must use ASL logging (which requires CGO) if running as root
    v2.4.5: Pulling from milvusdb/milvus
    4f4fb700ef54: Already exists
    838ea2fc0c5a: Download complete
    1f27396f6efc: Download complete
    5e94f6dcda58: Download complete
    fe556ec02776: Download complete
    ca3a09d8ea0c: Download complete
    25e4b36fd223: Download complete
    Digest: sha256:5a0a330981c925e53efe088a2864f27c33d3347868665ebdf0944c917bcb8c85
    Status: Downloaded newer image for milvusdb/milvus:v2.4.5
    Wait for Milvus Starting...
    Start successfully.
    To change the default Milvus configuration, add your settings to the user.yaml file and then restart the service.

```
3.2 Prepare the data for Milvus RAG sample (openAI-milvus-rag.py)
```
    wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
    unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
    
    wget https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip
    unzip -q -o reverse_image_search.zip

```

4. Run the Streamlit App
    The app needs openAI gpt-4 and Anthropic, so make sure you have api keys for both.
```bash
  poetry run streamlit run multi-llm.py
  poetry run python multi-llm.py
  poetry run python openAI-milvus-rag.py
```
