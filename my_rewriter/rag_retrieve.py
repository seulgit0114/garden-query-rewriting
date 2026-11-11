import logging
import sys
import os

# transformers 내부에서 logger.warning_once(DEPRECATION_MESSAGE, FutureWarning) 패턴을
# 사용하는데, Python logging이 DEPRECATION_MESSAGE % (FutureWarning,) 를 시도해
# TypeError: not all arguments converted during string formatting 가 발생한다.
# warning_once는 최초 1회만 실행되므로 첫 번째 임베딩 쿼리만 항상 실패하는 버그.
# transformers 로거를 ERROR 레벨로 올려 warning_once 자체를 차단한다.
logging.getLogger("transformers").setLevel(logging.ERROR)
from typing import Any, Dict, List, Optional, Sequence, Union, cast, Tuple
import typing as t
from collections import defaultdict
import json
import itertools

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    NodeWithScore
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import BaseRetriever

from my_rewriter.my_utils import achat
from rag.my_query_fusion_retriver import MyQueryFusionRetriever, FUSION_MODES
from rag.my_structure_retriever import MyStructureRetriever
from rag.prompts import STACKOVERFLOW_QA_PROMPT
from rag.gen_rewrites_from_rules import gen_rewrites_from_rules

# (Add) chromadb 강제종료
# Global ChromaDB client (reuse across queries to avoid connection issues)
_CHROMA_CLIENT = None

# 특정 인스턴스가 오직 하나만 존재하도록 보장하는 소프트웨어 설계 패턴
def get_chroma_client():
    """Get or create a singleton ChromaDB client"""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_db_path = os.path.join(project_root, "rag", "chroma_db")
        _CHROMA_CLIENT = chromadb.PersistentClient(path=chroma_db_path)
    return _CHROMA_CLIENT

def init_docstore() -> SimpleDocumentStore:
    docstore = SimpleDocumentStore()

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def read_docs(filename: str) -> Sequence[BaseNode]:
        doc_nodes = []
        full_path = os.path.join(project_root, filename) if not os.path.isabs(filename) else filename
        with open(full_path, 'r') as fin:
            for line in fin.readlines():
                obj = json.loads(line)
                myid = f'{obj["id"]}-{obj["answer_id"]}'
                doc = STACKOVERFLOW_QA_PROMPT.format(title=obj["question_title"], question_body=obj["question_body"], answer_body=obj["answer_body"])
                text_node = TextNode(
                    text=doc, 
                    id_ = myid
                )
                doc_nodes.append(text_node)
        return doc_nodes

    doc_nodes = read_docs('rag/stackoverflow-rewrite-query-optimization.jsonl')
    docstore.add_documents(doc_nodes)
    return docstore

def rag_retrieve(query: str, schema: str, docstore: SimpleDocumentStore, embed_dim: int,
                 RETRIEVER_TOP_K: int = 10, fast_llm=None) -> Dict:
    # Get shared ChromaDB client (singleton)
    db = get_chroma_client()

    # get collection
    stackoverflow_chroma_collection = db.get_or_create_collection("stackoverflow")

    # assign chroma as the vector_store to the context
    stackoverflow_vector_store = ChromaVectorStore(chroma_collection=stackoverflow_chroma_collection)
    stackoverflow_storage_context = StorageContext.from_defaults(vector_store=stackoverflow_vector_store)

    # load your index from stored vectors
    stackoverflow_index = VectorStoreIndex.from_vector_store(
        stackoverflow_vector_store, storage_context=stackoverflow_storage_context
    )

    stackoverflow_retriever = stackoverflow_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)

    retriever = MyQueryFusionRetriever(docstore=docstore, qa_retriever=stackoverflow_retriever,
                                       schema=schema, embed_dim=embed_dim, llm=fast_llm,
                                       mode=FUSION_MODES.RECIPROCAL_RANK,
                                       similarity_top_k=RETRIEVER_TOP_K,
                                       use_async=True, verbose=True)
    retriever_res = retriever.retrieve(query)
    logging.info('Retrieved Rewrite Cases: ' + str(retriever_res))
    return {"retriever_res": retriever_res, "rewrites": retriever._queries}

def rag_semantics_retrieve(query: str, schema: str, docstore: SimpleDocumentStore, RETRIEVER_TOP_K: int = 10) -> Dict:
    # Get shared ChromaDB client (singleton)
    db = get_chroma_client()

    # get collection
    stackoverflow_chroma_collection = db.get_or_create_collection("stackoverflow_summary")

    # assign chroma as the vector_store to the context
    stackoverflow_vector_store = ChromaVectorStore(chroma_collection=stackoverflow_chroma_collection)
    stackoverflow_storage_context = StorageContext.from_defaults(vector_store=stackoverflow_vector_store)

    # load your index from stored vectors
    stackoverflow_index = VectorStoreIndex.from_vector_store(
        stackoverflow_vector_store, storage_context=stackoverflow_storage_context
    )

    retriever = stackoverflow_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)
    result = retriever.retrieve(query)

    node_with_scores = []
    for cur_node in result:
        cur_node_ids = eval(cur_node.metadata['references'])
        assert len(cur_node_ids) == 1
        cur_node_id = cur_node_ids[0]
        node_with_scores.append(NodeWithScore(node=docstore.get_node(cur_node_id), score=cur_node.get_score()))
    logging.info('Retrieved Rewrite Cases: ' + str(node_with_scores))

    rewrites, _ = gen_rewrites_from_rules(sql=query, schema=schema, fun=achat, verbose=True)
    return {"retriever_res": node_with_scores, "rewrites": rewrites}

def rag_structure_retrieve(query: str, schema: str, docstore: SimpleDocumentStore, embed_dim: int, RETRIEVER_TOP_K: int = 10) -> Dict:
    # Get shared ChromaDB client (singleton)
    db = get_chroma_client()

    # get collection
    stackoverflow_chroma_collection = db.get_or_create_collection("stackoverflow_structure")

    # assign chroma as the vector_store to the context
    stackoverflow_vector_store = ChromaVectorStore(chroma_collection=stackoverflow_chroma_collection)
    stackoverflow_storage_context = StorageContext.from_defaults(vector_store=stackoverflow_vector_store)

    # load your index from stored vectors
    stackoverflow_index = VectorStoreIndex.from_vector_store(
        stackoverflow_vector_store, storage_context=stackoverflow_storage_context
    )

    stackoverflow_retriever = stackoverflow_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)

    retriever = MyStructureRetriever(docstore=docstore, qa_retriever=stackoverflow_retriever, embed_dim=embed_dim, schema=schema, mode=FUSION_MODES.RECIPROCAL_RANK, similarity_top_k=RETRIEVER_TOP_K, use_async=False, verbose=True)
    retriever_res = retriever.retrieve(query)
    logging.info('Retrieved Rewrite Cases: ' + str(retriever_res))
    return {"retriever_res": retriever_res, "rewrites": retriever._queries}
