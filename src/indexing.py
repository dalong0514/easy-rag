import weaviate
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.classes.config import ConsistencyLevel
from llama_index.core import StorageContext

Settings.embed_model = resolve_embed_model("local:/Users/Daglas/dalong.modelsets/bge-m3")

def create_document_index(input_files, index_name, batch_size):
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()
    # 创建集合
    documents = client.collections.create(name=index_name)
    print("documents collection has been created.")
    # load documents
    documents = SimpleDirectoryReader(input_files=input_files).load_data()

    custom_batch = client.batch.fixed_size(
        batch_size=batch_size,
        concurrent_requests=4,
        consistency_level=ConsistencyLevel.ALL,
    )
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name,
        # we pass our custom batch as a client_kwargs
        client_kwargs={"custom_batch": custom_batch},
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    print("All vector data has been written to Weaviate.")

    client.close()  # Free up resources


def create_document_collection():
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()
    
    # 创建集合
    documents = client.collections.create(name="AgentRAGDocument")
    print("documents collection has been created.")

    client.close()  # Free up resources

def import_vector_document():
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()

    # load documents
    documents = SimpleDirectoryReader(input_files=["../data/papers/metagpt.pdf"]).load_data()

    custom_batch = client.batch.fixed_size(
        batch_size=1024,
        concurrent_requests=4,
        consistency_level=ConsistencyLevel.ALL,
    )
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name="AgentRAGDocument",
        # we pass our custom batch as a client_kwargs
        client_kwargs={"custom_batch": custom_batch},
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    print("All vector data has been written to Weaviate.")

    client.close()  # Free up resources