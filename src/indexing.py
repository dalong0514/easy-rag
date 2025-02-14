import weaviate
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.weaviate import WeaviateVectorStore
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

def create_document_index(input_files, index_name, chunk_size=1024, chunk_overlap=200):
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
        # 设置文档分割器
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=index_name,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True  #显示进度
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