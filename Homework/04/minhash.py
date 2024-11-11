import re
import pandas as pd
import numpy as np


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+"," ",text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)      
        return set(text.split(' '))
    
    def get_occurrence_matrix(self, corpus_of_texts: list[str]) -> pd.DataFrame:
        '''
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        '''
        # TODO:
        tokenized_texts = [self.tokenize(text) for text in corpus_of_texts]
        tokens = sorted(set().union(*tokenized_texts))
        data = {i: [1 if token in doc else 0 for token in tokens] for i, doc in enumerate(tokenized_texts)}
        df = pd.DataFrame(data, index=tokens)

        df.sort_index(inplace=True)
        return df
    
    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a
    
    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        '''
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        '''
        values_dict = {
            'a': [3, 4, 5, 7, 8],
            'b': [3, 4, 5, 7, 8] 
        }
        a = values_dict['a'][permutation_index]
        b = values_dict['b'][permutation_index]
        return (a*(x+1) + b) % prime_num_rows 
    
    
    def get_minhash_similarity(self, array_a: np.array, array_b: np.array) -> float:
        '''
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        '''
        # TODO:
        return np.mean(array_a == array_b)

    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        pairs = []
        num_docs = min_hash_matrix.shape[1]
        for i in range(num_docs):
            for j in range(i+1, num_docs):
                similarity = self.get_minhash_similarity(min_hash_matrix[:, i], min_hash_matrix[:, j])
                if similarity > self.threshold:
                    pairs.append((i, j))
        return pairs
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO: 
        num_docs = min_hash_matrix.shape[1]
        similarity_matrix = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for j in range(i, num_docs):
                similarity = self.get_minhash_similarity(min_hash_matrix[:, i], min_hash_matrix[:, j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        return similarity_matrix
     
    
    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы. 

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        '''
        # TODO:
        num_tokens, num_docs = occurrence_matrix.shape
        prime_num_rows = num_tokens
        while not self.is_prime(prime_num_rows):
            prime_num_rows += 1
        minhash_matrix = np.full((self.num_permutations, num_docs), np.inf)
        for perm_idx in range(self.num_permutations):
            for row in range(num_tokens):
                perm_index = self.get_new_index(row, perm_idx, prime_num_rows)
                for doc_idx, has_token in enumerate(occurrence_matrix.iloc[row]):
                    if has_token and perm_index < minhash_matrix[perm_idx, doc_idx]:
                        minhash_matrix[perm_idx, doc_idx] = perm_index
        return minhash_matrix

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs

class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold
    
    
    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        '''
        Вовзращает расстояние Жаккарда для двух сетов. 
        '''
        # TODO:
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union

    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        num_docs = min_hash_matrix.shape[1]
        similar_pairs = []
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
               similarity = np.mean(min_hash_matrix[:, i] == min_hash_matrix[:, j])
               if similarity >= self.threshold:
                    similar_pairs.append((i, j))
        return similar_pairs
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO: 
        num_docs = min_hash_matrix.shape[1]
        similarity_matrix = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                similarity = np.mean(min_hash_matrix[:, i] == min_hash_matrix[:, j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        return similarity_matrix
     
    
    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2
        
        '''
        # TODO:
        num_tokens, num_docs = occurrence_matrix.shape
        prime_num_rows = num_tokens
        while not self.is_prime(prime_num_rows):
            prime_num_rows += 1
        minhash_matrix = np.full((self.num_permutations, num_docs), np.inf)
        for perm_idx in range(self.num_permutations):
            for row in range(num_tokens):
                for doc_idx, has_token in enumerate(occurrence_matrix.iloc[row]):
                    if has_token:
                        minhash_matrix[perm_idx, doc_idx] = min(minhash_matrix[perm_idx, doc_idx], row)
        minhash_matrix = minhash_matrix.astype(int)
        return minhash_matrix

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash_jaccard(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs

    
    
