import os
import weaviate
from llama_index.core import SimpleDirectoryReader, StorageContext, ServiceContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, SentenceWindowNodeParser, get_leaf_nodes
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.vector_stores.weaviate import WeaviateVectorStore

Settings.embed_model = resolve_embed_model("local:/Users/Daglas/dalong.modelsets/bge-m3")

def build_basic_fixed_size_index(input_files, index_name, chunk_size=1024, chunk_overlap=200):
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

def delete_document_collection(index_name):
    """删除 Weaviate 中的集合"""
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()
    
    # 删除集合
    client.collections.delete(index_name)

    print("documents collection has been deleted.")
    
    client.close()  # Free up resources

if __name__ == "__main__":
    delete_document_collection("xx")