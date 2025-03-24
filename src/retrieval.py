import weaviate, os
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.vector_stores.weaviate import WeaviateVectorStore

reranker_model_name = "/Users/Daglas/dalong.modelsets/bge-reranker-v2-m3"

def get_all_index_names():
    """获取Weaviate中所有可用的索引名称
    
    连接到本地Weaviate实例，检索并返回所有集合名称。
    
    Returns:
        list: 按字母顺序排序的索引名称列表
        
    Raises:
        Exception: 连接或检索过程中可能出现的异常
    """
    try:
        client = weaviate.connect_to_local()
        # 直接获取集合名称列表（字符串列表）
        collections = client.collections.list_all()  # 直接返回字符串列表
        print("Available index names in Weaviate:")
        names = []
        for name in collections:
            names.append(name)
        names.sort()  # 按字母顺序排序
        return names
    except Exception as e:
        print(f"Error listing index names: {str(e)}")
        return []
    finally:
        if 'client' in locals():
            client.close()

def basic_query_from_documents(question, index_names, similarity_top_k):
    """从多个索引中检索相关文档片段
    
    根据问题从多个Weaviate索引中检索相关内容，通过组合检索结果并按相关性排序。
    
    Args:
        question (str): 用户查询问题
        index_names (str|list): 要查询的索引名称，可以是单个字符串或字符串列表
        similarity_top_k (int): 检索的最大结果数量
        
    Returns:
        list: 检索到的源节点列表，按相关性分数排序
        
    Raises:
        Exception: 检索过程中可能出现的异常
    """
    try:
        client = weaviate.connect_to_local()
        
        if isinstance(index_names, str):
            index_names = [index_names]
            
        vector_indices = []
        for index_name in index_names:
            vector_store = WeaviateVectorStore(
                weaviate_client=client, 
                index_name=index_name
            )
            vector_index = VectorStoreIndex.from_vector_store(vector_store)
            vector_indices.append(vector_index)
        
        # 创建每个索引的检索器
        retrievers = [index.as_retriever(similarity_top_k=similarity_top_k) for index in vector_indices]
        
        # 自定义复合检索器
        # 修改后的 MultiIndexRetriever 实现
        class MultiIndexRetriever(BaseRetriever):
            """复合检索器，用于从多个索引中检索并合并结果
            
            继承自BaseRetriever，组合多个检索器的结果并按相关性排序。
            
            Attributes:
                retrievers (list): 检索器列表
                similarity_top_k (int): 要返回的最大结果数
            """
            def __init__(self, retrievers, similarity_top_k):
                """初始化复合检索器
                
                Args:
                    retrievers (list): 检索器列表
                    similarity_top_k (int): 要返回的最大结果数
                """
                super().__init__()
                self.retrievers = retrievers
                self.similarity_top_k = similarity_top_k  # 存储全局 top_k 值

            def _retrieve(self, query, **kwargs):
                """实际执行检索的内部方法
                
                Args:
                    query (str): 查询字符串
                    **kwargs: 额外的关键字参数
                    
                Returns:
                    list: 合并后并排序的节点列表，限制为top_k个
                """
                all_nodes = []
                # 收集所有检索器的节点
                for retriever in self.retrievers:
                    retrieved_nodes = retriever.retrieve(query, **kwargs)
                    all_nodes.extend(retrieved_nodes)
                
                # 按相似度分数降序排序（分数越高越相关）
                sorted_nodes = sorted(all_nodes, key=lambda x: x.score, reverse=True)
                
                # 截取前 similarity_top_k 个节点
                return sorted_nodes[:self.similarity_top_k]
                # top_k_nodes = sorted_nodes[:self.similarity_top_k]
                
                # # 剔除score值低于0.4的节点
                # filtered_nodes = [node for node in top_k_nodes if node.score >= 0.4]
                
                # return filtered_nodes

        # 创建复合检索器时传入 similarity_top_k 参数
        combined_retriever = MultiIndexRetriever(retrievers, similarity_top_k=similarity_top_k)  # 关键修改

        # 创建查询引擎时不需要再设置 similarity_top_k（已在检索器层处理）
        query_engine = RetrieverQueryEngine.from_args(combined_retriever)
        
        response = query_engine.query(question)

        return response.source_nodes
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()
            print("\nWeaviate connection closed.")

def basic_query_from_documents_for_one_collection(question, index_name, similarity_top_k):
    """从单个索引中检索相关文档片段
    
    根据问题从指定的单个Weaviate索引中检索相关内容。
    
    Args:
        question (str): 用户查询问题
        index_name (str): 要查询的索引名称
        similarity_top_k (int): 检索的最大结果数量
        
    Returns:
        list: 检索到的源节点列表
        
    Raises:
        Exception: 检索过程中可能出现的异常
    """
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=index_name
        )

        vector_index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine = vector_index.as_query_engine(similarity_top_k=similarity_top_k)

        response = query_engine.query(question)

        return response.source_nodes

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("\nWeaviate connection closed.")


def sentence_window_query_from_documents(
    question,
    index_name,
    similarity_top_k=12,
    rerank_top_n=4,
):
    """使用句子窗口策略从索引中检索文档片段
    
    利用句子窗口技术从指定索引检索相关内容，并使用重排序器提高结果相关性。
    
    Args:
        question (str): 用户查询问题
        index_name (str): 要查询的索引名称
        similarity_top_k (int, optional): 初始检索的最大结果数量，默认为12
        rerank_top_n (int, optional): 重排序后保留的结果数量，默认为4
        
    Returns:
        list: 检索并重排序后的源节点列表
        
    Raises:
        Exception: 检索过程中可能出现的异常
    """
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=index_name
        )

        sentence_index = VectorStoreIndex.from_vector_store(vector_store)

        # define postprocessors
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model=reranker_model_name
        )

        sentence_window_engine = sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
        )

        response = sentence_window_engine.query(question)

        return response.source_nodes

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("\nWeaviate connection closed.")

def automerging_query_from_documents(
    question,
    index_name,
    similarity_top_k=12,
    rerank_top_n=2,
):
    """使用自动合并检索从索引中检索文档片段
    
    利用自动合并检索器从分层索引中智能检索内容，并使用重排序器提高结果相关性。
    
    Args:
        question (str): 用户查询问题
        index_name (str): 要查询的索引名称
        similarity_top_k (int, optional): 初始检索的最大结果数量，默认为12
        rerank_top_n (int, optional): 重排序后保留的结果数量，默认为2
        
    Returns:
        list: 经过自动合并检索和重排序后的源节点列表
        
    Raises:
        Exception: 检索过程中可能出现的异常
    """
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=index_name
        )

        automerging_index = VectorStoreIndex.from_vector_store(vector_store)

        base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)

        retriever = AutoMergingRetriever(
            base_retriever, 
            automerging_index.storage_context, 
            verbose=True
        )

        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model=reranker_model_name
        )

        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, 
            node_postprocessors=[rerank]
        )

        response = auto_merging_engine.query(question)

        return response.source_nodes

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("\nWeaviate connection closed.")
        # query_engine = automerging_index.as_query_engine(similarity_top_k=similarity_top_k)
        # response = query_engine.query(question)
        # # print(response.source_nodes)
        # with open("/Users/Daglas/Downloads/output.json", 'a', encoding='utf-8') as file:
        #     file.write(str(response.source_nodes) + '\n\n')


if __name__ == "__main__":
    get_all_index_names()