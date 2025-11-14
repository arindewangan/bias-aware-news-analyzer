import uuid
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_client = None
_collection = None
_embedding = None
_st_model = None

def ensure_collection(name="summaries"):
    global _client, _collection, _embedding
    if _client is None:
        _client = chromadb.Client()
    if _embedding is None:
        try:
            _embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception:
            _embedding = None
            try:
                from sentence_transformers import SentenceTransformer
                global _st_model
                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                _st_model = None
    if _collection is None:
        if _embedding is not None:
            _collection = _client.get_or_create_collection(name=name, embedding_function=_embedding)
        else:
            _collection = _client.get_or_create_collection(name=name)
    return _collection

def _get_embeddings(texts):
    try:
        if _embedding is not None:
            return _embedding(texts)
    except Exception:
        pass
    try:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            globals()["_st_model"] = SentenceTransformer("all-MiniLM-L6-v2")
        return _st_model.encode(texts)
    except Exception:
        return None

def add_document(text, metadata=None, doc_id=None):
    col = ensure_collection()
    did = doc_id or str(uuid.uuid4())
    try:
        col.add(documents=[text], metadatas=[metadata or {}], ids=[did])
        return did
    except Exception:
        embs = _get_embeddings([text])
        if embs is None:
            return None
        try:
            col.add(documents=[text], embeddings=embs, metadatas=[metadata or {}], ids=[did])
            return did
        except Exception:
            return None

def add_documents(texts, metadatas=None, ids=None):
    col = ensure_collection()
    mds = metadatas or [{} for _ in texts]
    ids = ids or [str(uuid.uuid4()) for _ in texts]
    try:
        col.add(documents=texts, metadatas=mds, ids=ids)
        return ids
    except Exception:
        embs = _get_embeddings(texts)
        if embs is None:
            return []
        try:
            col.add(documents=texts, embeddings=embs, metadatas=mds, ids=ids)
            return ids
        except Exception:
            return []

def delete_documents(ids):
    col = ensure_collection()
    try:
        col.delete(ids=ids)
    except Exception:
        pass

def query(text, n_results=3, where=None):
    col = ensure_collection()
    try:
        return col.query(query_texts=[text], n_results=n_results, where=where)
    except Exception:
        embs = _get_embeddings([text])
        if embs is None:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        try:
            return col.query(query_embeddings=embs, n_results=n_results, where=where)
        except Exception:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}