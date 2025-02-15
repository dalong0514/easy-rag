import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceWindowNodeParser, get_leaf_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.core.retrievers import AutoMergingRetriever
import numpy as np

# 修改后的trulens导入方式
from trulens_eval.feedback import Feedback
from trulens.feedback import GroundTruthAgreement
from trulens.providers.openai import OpenAI
from trulens.apps.llamaindex import TruLlama

from helper import get_api_key
# from langchain_openai import OpenAI
# 在执行代码前手动忽略这些警告
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

api_key = get_api_key()
base_url= "https://api.302.ai/v1"
model_name = "deepseek-r1-aliyun"
embed_model_name = "local:/Users/Daglas/dalong.modelsets/bge-m3"

openai = OpenAI()

# openai = OpenAI(
#     base_url=base_url,
#     api_key=api_key,
#     model_name=model_name,
#     streaming=True
# )

qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

ground_truth_text = [
    {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
    {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
]
grounded = GroundTruthAgreement(ground_truth=ground_truth_text, provider=openai)
groundedness = Feedback(grounded.agreement_measure, name="Ground Truth")

feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder



def build_sentence_window_index(
    document, llm, embed_model=embed_model_name, save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="/Users/Daglas/dalong.modelsets/bge-reranker-v2-m3"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


# for auto-merging retriever
def build_automerging_index(
    documents,
    llm,
    embed_model=embed_model_name,
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="/Users/Daglas/dalong.modelsets/bge-reranker-v2-m3"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine