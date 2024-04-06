import numpy as np
import faiss

class NearestNeighbors:
    """
    Class for NearestNeighbors.   
    """
    def __init__(self, n_neighbors=10, metric='cosine', rerank_from=-1):
        """
         metric = 'cosine' / 'binary' 
         if metric ~= 'cosine' and rerank_from > n_neighbors then a cosine rerank will be performed
        """
        self.n_neighbors = n_neighbors
        self.metric = metric        
        self.rerank_from = rerank_from                
        
    def normalize(self, a):
        return a / np.sum(a**2, axis=1, keepdims=True)
    
    def fit(self, data, o_data=None):
        if self.metric == 'cosine':
            data = self.normalize(data)
            self.index = faiss.IndexFlatIP(data.shape[1])     
        elif self.metric == 'binary':
            self.o_data = data if o_data is None else o_data
            #assuming data already packed
            self.index = faiss.IndexBinaryFlat(data.shape[1]*8)            
        self.index.add(np.ascontiguousarray(data))
        
    def kneighbors(self, q_data):                
        if self.metric == 'cosine':
            q_data = self.normalize(q_data)      
            sim, idx = self.index.search(q_data, self.n_neighbors)        
        else:            
            if self.metric == 'binary':
                print('This is binary search.')
                bq_data = np.packbits((q_data > 0.0).astype(bool), axis=1)
            sim, idx = self.index.search(bq_data, max(self.rerank_from, self.n_neighbors))
            
            if self.rerank_from > self.n_neighbors:
                re_sims = np.zeros([len(q_data), self.n_neighbors], dtype=float)
                re_idxs = np.zeros([len(q_data), self.n_neighbors], dtype=float)
                for i, q in enumerate(q_data):
                    rerank_data = self.o_data[idx[i]]
                    rerank_search = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine') 
                    rerank_search.fit(rerank_data)
                    re_sim, re_idx = rerank_search.kneighbors(np.asarray([q]))
                    print("re_idx: ", re_idx)
                    re_sims[i, :] = re_sim
                    re_idxs[i, :] = idx[i][re_idx]
                idx = re_idxs
                sim = re_sims

        return sim, idx