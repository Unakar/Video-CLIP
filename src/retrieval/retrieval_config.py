# set dataset path
DATA_PATH = '/home/group2024-xietian/CLIP4Clip/clip4clip-webvid150k/data'
VISION_EMBEDDINGS_FILE_COSINE = DATA_PATH + '/dataset_v1_visual_features.npy'
VISION_EMBEDDINGS_FILE_BINARY = DATA_PATH + '/dataset_v1_visual_features_binary_packed.npy'
VIDEO_CSV_PATH = DATA_PATH + '/dataset_v1.csv'
TOPK = 10

# database milvus settings
MILVUS_DB_NAME = "milvus_video"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT =  19530
VECTOR_DIMENSION = 2048
METRIC_TYPE = "COSINE"
MILVUS_TABLE = "similarity_search"
TOP_K = 10
INDEX_TYPE = "IVF_FLAT"
DISTANCE_THERSHOLD =  0.5

## clip model path
CLIP_MODEL = "/home/group2024-xietian/CLIP4Clip/clip4clip-webvid150k"
CLIP_TOKENIZER = "/home/group2024-xietian/CLIP4Clip/clip4clip-webvid150k"