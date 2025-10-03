from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.utils.env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL

multiModal_llm = ChatOpenAI(  # 多模态大模型
    model='qwen-vl-plus',
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL,
)
class CustomQwen3Embeddings(Embeddings):
    """自定义一个qwen3的Embedding和langchain整合的类"""


    def __init__(self, model_name):
        self.qwen3_embedding = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.qwen3_embedding.encode(texts)