import pandas as pd
import numpy as np


from minhash import MinHash

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold
        
    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        # TODO:
        num_rows = minhash.shape[0]
        if self.num_buckets > num_rows:
            self.num_buckets = num_rows
        rows_per_bucket = num_rows // self.num_buckets
        extra_rows = num_rows % self.num_buckets
        buckets = []
        current_row = 0
        for i in range(extra_rows):
            end_row = current_row + rows_per_bucket +1
            bucket = minhash[current_row:end_row, :]
            buckets.append(bucket)
            current_row = end_row

        for i in range(self.num_buckets - extra_rows):
            end_row = current_row + rows_per_bucket
            bucket = minhash[current_row:end_row, :]
            buckets.append(bucket)
            current_row = end_row
        buckets = np.array(buckets, dtype=object)
        return buckets
    
    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандижатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''
        # TODO:
        similar_candidates= []
        for bucket in buckets:
            num_docs = bucket.shape[1]
            seen_hashes = {}
            for doc_idx in range(num_docs):
                hash_value = tuple(bucket[:, doc_idx])
                if hash_value in seen_hashes:
                    for prev_doc_idx in seen_hashes[hash_value]:
                        similar_candidates.append((prev_doc_idx, doc_idx))                
                if hash_value not in seen_hashes:
                    seen_hashes[hash_value] = []
                seen_hashes[hash_value].append(doc_idx)
        return similar_candidates
        
    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets)
        
        return set(similar_candidates)
    
