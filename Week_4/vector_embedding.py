from langchain_community.embeddings.ollama import OllamaEmbeddings

def embedding_function():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings