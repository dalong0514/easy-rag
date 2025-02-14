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

def get_all_files_from_directory(directory_path):
    """获取指定目录下的所有文件路径
    
    Args:
        directory_path (str): 目录路径
    
    Returns:
        list: 包含所有文件路径的列表
    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    return [str(file) for file in path.glob("*") if file.is_file()]

def create_document_index(input_files, index_name):
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        # 检查集合是否存在，如果存在则删除
        if client.collections.exists(index_name):
            client.collections.delete(index_name)
            print(f"Existing collection {index_name} has been deleted.")
        
        # 创建集合
        documents = client.collections.create(name=index_name)
        print("documents collection has been created.")

        # load documents
        documents = SimpleDirectoryReader(input_files=input_files).load_data()

        custom_batch = client.batch.fixed_size(
            batch_size=150,
            concurrent_requests=3,
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
            documents, 
            storage_context=storage_context,
            show_progress=True  # Add progress visualization
        )

        print("All vector data has been written to Weaviate.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed
            print("Weaviate connection closed.")

def delete_document_collection(index_name):
    """删除 Weaviate 中的集合"""
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()
    
    # 删除集合
    client.collections.delete(index_name)

    print("documents collection has been deleted.")
    
    client.close()  # Free up resources