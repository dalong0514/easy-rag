import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weaviate
from llama_index.core import SimpleDirectoryReader, StorageContext, ServiceContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, SentenceWindowNodeParser, get_leaf_nodes
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from src.utils import get_all_files_from_directory

Settings.embed_model = resolve_embed_model("local:/Users/Daglas/dalong.modelsets/bge-m3")

def build_basic_fixed_size_index(input_files, index_name, chunk_size=1024, chunk_overlap=200):
    """构建基础固定大小的索引
    
    使用简单的文档分割策略创建向量索引，将文档分割成固定大小的块。
    
    Args:
        input_files (list): 输入文件路径列表
        index_name (str): 索引名称，将作为Weaviate集合名
        chunk_size (int, optional): 文档分块大小，默认为1024个字符
        chunk_overlap (int, optional): 相邻块之间的重叠字符数，默认为200
        
    Returns:
        None: 函数无返回值，但会在Weaviate中创建指定名称的集合和索引
        
    Raises:
        Exception: 在处理过程中可能出现的各种异常
    """
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

# for auto-merging retriever
def build_automerging_index(input_files, index_name, chunk_sizes=None):
    """构建自动合并索引
    
    使用层次化节点解析器创建向量索引，支持多尺度文档分割，适用于自动合并检索。
    
    Args:
        input_files (list): 输入文件路径列表
        index_name (str): 索引名称，将作为Weaviate集合名
        chunk_sizes (list, optional): 文档分块大小列表，默认为[2048, 512, 128]
        
    Returns:
        None: 函数无返回值，但会在Weaviate中创建指定名称的集合和索引
        
    Raises:
        Exception: 在处理过程中可能出现的各种异常
    """
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

        chunk_sizes = chunk_sizes or [2048, 512, 128]
        # load documents
        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

        # 获取所有节点和叶子节点
        nodes = node_parser.get_nodes_from_documents(documents)
        # leaf_nodes = get_leaf_nodes(nodes)

        # 初始化 Weaviate 向量存储
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=index_name,
        )

        # 创建存储上下文，并将所有节点（包括父节点）添加到 docstore 中
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 构建索引时传入叶子节点，同时启用 store_nodes_override，确保索引使用 docstore 中的完整节点信息
        automerging_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            store_nodes_override=True,
            show_progress=True
        )

        print("All vector data has been written to Weaviate.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed
            print("Weaviate connection closed.")


# the sentence window retrieval
def build_sentence_window_index(input_files, index_name):
    """构建句子窗口索引
    
    使用句子窗口节点解析器创建向量索引，对每个句子保留上下文窗口，适用于上下文感知检索。
    
    Args:
        input_files (list): 输入文件路径列表
        index_name (str): 索引名称，将作为Weaviate集合名
        
    Returns:
        None: 函数无返回值，但会在Weaviate中创建指定名称的集合和索引
        
    Raises:
        Exception: 在处理过程中可能出现的各种异常
    """
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
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=5,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        # 获取所有节点和叶子节点
        nodes = node_parser.get_nodes_from_documents(documents)

        # 初始化 Weaviate 向量存储
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=index_name,
        )

        # 创建存储上下文
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 构建索引时传入数据
        sentence_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )

        print("All vector data has been written to Weaviate.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed
            print("Weaviate connection closed.")

def delete_document_collections(index_names):
    """批量删除 Weaviate 中的集合
    
    Args:
        index_names (str|list): 要删除的索引名称，可以是单个字符串或字符串列表
        
    Returns:
        None: 函数无返回值，但会删除Weaviate中的指定集合
        
    Raises:
        Exception: 在处理过程中可能出现的各种异常
    """
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()
    
    try:
        # 如果传入的是单个字符串，转换为列表
        if isinstance(index_names, str):
            index_names = [index_names]
            
        # 遍历并删除每个集合
        for index_name in index_names:
            if client.collections.exists(index_name):
                client.collections.delete(index_name)
                print(f"Collection '{index_name}' has been deleted.")
            else:
                print(f"Collection '{index_name}' does not exist.")
                
    except Exception as e:
        print(f"Error occurred while deleting collections: {str(e)}")
        raise
    finally:
        client.close()  # 释放资源
        print("Weaviate connection closed.")

if __name__ == "__main__":
    input_files = get_all_files_from_directory(
        "/Users/Daglas/dalong.github/dalong.readnotes/20240101复制书籍/2024097深度学习入门", 
        "md")
    build_basic_fixed_size_index(input_files, "Book2024097Deep_Learning_from_Scratch", )
    # build_basic_fixed_size_index(["/Users/Daglas/dalong.processdesign/03规范汇总/GBT20801-2020压力管道规范.md"], "Standard_GBT20801_2020", )