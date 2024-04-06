from .NearstNeighbors import NearestNeighbors
from .video_database import VideoDataBase
from .retrieval_config import CLIP_TOKENIZER,CLIP_MODEL
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

class retrieval_pipeline:
    def __init__(self):
        self.model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL)
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER)
        self.video_database = VideoDataBase()
        self.search = NearestNeighbors(n_neighbors=5, metric='binary', rerank_from=100)
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