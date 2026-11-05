try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
from src.retrievers import vector_retriever, bm25_retriever

TOP_K = 3

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
    id_key="id")          # дедупликация по id фрагмента

retriever = ensemble | RunnableLambda(lambda docs: docs[:TOP_K])
