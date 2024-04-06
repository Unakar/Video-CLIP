from .NearstNeighbors import NearestNeighbors
from .video_database import VideoDataBase
import os
import numpy as np
import pandas as pd
from IPython import display
import faiss
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

class retrieval_pipeline:
    def __init__(self):
        self.model = CLIPTextModelWithProjection.from_pretrained("/home/group2024-xietian/CLIP4Clip/clip4clip-webvid150k")
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/group2024-xietian/CLIP4Clip/clip4clip-webvid150k")
        self.video_database = VideoDataBase()
        self.search = NearestNeighbors(n_neighbors=10, metric='cosine', rerank_from=-1)
        self.search.fit(self.video_database.database_binary,self.video_database.database)

    def retrieval(self,search_sentence):
        inputs = self.tokenizer(text=search_sentence , return_tensors="pt")
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
        # Normalizing the embeddings:
        final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
        sequence_output = final_output.cpu().detach().numpy()
        
        sims, idxs = self.search.kneighbors(sequence_output)    
        urls = self.video_database.database_df.iloc[idxs[0]]['contentUrl'].to_list()
        AUTOPLAY_VIDEOS = []
        for url in urls:
            AUTOPLAY_VIDEOS.append("""<video controls muted autoplay>
                    <source src={} type="video/mp4">
                    </video>""".format(url))
        return AUTOPLAY_VIDEOS