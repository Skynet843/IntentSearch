from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import os
import json


class ProductSearchIndexer:
    def __init__(self, index_path="data/index.faiss", id_map_path="data/id_map.json",
                 product_model_name='all-MiniLM-L6-v2',
                 query_model_name='multi-qa-MiniLM-L6-cos-v1',
                 reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.product_model_name = product_model_name
        self.query_model_name = query_model_name
        self.reranker_model_name = reranker_model_name
        self.dimension = 384

        # Load models
        self.product_model = SentenceTransformer(self.product_model_name)
        self.query_model = SentenceTransformer(self.query_model_name)
        self.reranker = CrossEncoder(self.reranker_model_name)

        # Load or initialize FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors.")
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            print("⚠️ Created new FAISS index.")

        # Load or initialize ID map
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, "r") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = []

        # Internal dictionary to track product_id → full text (for reranking)
        self.product_id_to_text = {}

    def append(self, product_tuples):
        if not product_tuples:
            print("⚠️ No products to append.")
            return

        product_texts = [text for _, text in product_tuples]
        product_ids = [pid for pid, _ in product_tuples]

        # Store product text for reranker use
        for pid, text in product_tuples:
            self.product_id_to_text[pid] = text

        embeddings = self.product_model.encode(product_texts, normalize_embeddings=True).astype('float32')
        self.index.add(embeddings)
        self.id_map.extend(product_ids)

        self._save_index()
        self._save_id_map()

        print(f"✅ Appended {len(product_tuples)} products. Total in index: {self.index.ntotal}")

    def search(self, query, top_k=20, return_scores=False, rerank=True):
        """
        Search and optionally rerank results using a cross-encoder.

        Args:
            query (str): The user query.
            top_k (int): Number of results to return.
            return_scores (bool): Whether to return similarity scores.
            rerank (bool): Whether to rerank with cross-encoder.

        Returns:
            List[str] or List[Tuple[str, float]]
        """
        if self.index.ntotal == 0:
            print("⚠️ Index is empty. Add products before searching.")
            return []

        # Step 1: FAISS search
        query_vec = self.query_model.encode([query], normalize_embeddings=True).astype('float32')
        D, I = self.index.search(query_vec, top_k)

        candidates = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.id_map):
                pid = self.id_map[idx]
                text = self.product_id_to_text.get(pid, "")
                candidates.append((pid, text, score))

        if not rerank:
            return [(pid, score) if return_scores else pid for pid, _, score in candidates]

        # Step 2: Rerank using cross-encoder
        rerank_inputs = [(query, text) for _, text, _ in candidates]
        rerank_scores = self.reranker.predict(rerank_inputs)

        # Combine and sort
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        results = []
        for ((pid, _, _), score) in reranked:
            if return_scores:
                results.append((pid, float(score)))
            else:
                results.append(pid)

        return results

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def _save_id_map(self):
        with open(self.id_map_path, "w") as f:
            json.dump(self.id_map, f)

    def get_index_size(self):
        return self.index.ntotal
