from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    #embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    #)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    #embeddings = OllamaEmbeddings(model="voyage-lite-02-instruct", num_gpu=1)
    #embeddings = OllamaEmbeddings(model="llama3", num_gpu=1)
    
    return embeddings
