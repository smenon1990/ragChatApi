import faiss
import pickle
import os
import numpy as np

INDEX_PATH = "/data/faiss.index"
CHUNKS_PATH = "/data/chunks.pkl"


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim

        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.chunks = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(texts)
        self._persist()

    def search(self, embedding, k=5):
        distances, indices = self.index.search(
            np.array([embedding]).astype("float32"), k
        )
        return [self.chunks[i] for i in indices[0]]

    def _persist(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
