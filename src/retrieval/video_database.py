'''
Offer 2 different ways to deal with video data:
1. using pre-computed visual features (cosine similarity / binary search)
2. vector-database (milvus) for new video to insert and delete
'''
import os
from retrieval_config import *
import numpy as np
import pandas as pd
class VideoDataBase:
    def __init__(self):
        self.database = np.load(VISION_EMBEDDINGS_FILE_COSINE, mmap_mode='r')
        self.database_binary = np.load(VISION_EMBEDDINGS_FILE_BINARY)
        self.database_df = pd.read_csv(VIDEO_CSV_PATH)
    def milvus_helpers(self):
        pass
    
        