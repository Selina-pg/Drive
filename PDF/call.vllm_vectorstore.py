import logging
import json
import requests
import numpy as np
from vanna.vllm import Vllm
from pymilvus import MilvusClient
from dataclasses import dataclass
from vanna.milvus import Milvus_VectorStore


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MilvusConfig:
    """Milvus configuration data class"""
    actual_db: str
    uri: str = "http://10.13.18.40:19530"  
    db_prefix: str = "SQLRAG_" 

MILVUS_CONFIGS = {
    "SQLRAG_ALS": MilvusConfig(
        actual_db="SQLRAG_ALS",
        uri="http://10.13.18.40:19530",
        db_prefix="SQLRAG_"
    )
}

# ---- Embedding API ----
class CustomEmbeddingFunction:
    def __init__(self, api_url="http://10.13.18.40:14514/embed", embedding_model="Conan-embedding-v1"):
        self.api_url = api_url
        self.embedding_model = embedding_model
        # 測試 API
        test_response = self._call_api(["Test connection"])
        if test_response is not None:
            logging.info(f"CustomEmbeddingFunction initialized with API at: {api_url}")
        else:
            raise ConnectionError("Failed to connect to embedding API")

    def _call_api(self, texts):
        payload = {
            "text": texts,
            "embedding_model": self.embedding_model
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        embeddings = np.array(result.get("embeddings", []))
        return embeddings

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self._call_api(texts)

    def encode_documents(self, texts):
        return self.__call__(texts)

    def encode_queries(self, texts):
        return self.__call__(texts)
    

# ---- 替換的 Embedding API ----
class CustomEmbeddingFunction:
    def __init__(self, api_url="http://10.13.18.40:14514/embed", embedding_model="Conan-embedding-v1", test_on_init=True, timeout=10):
        self.api_url = api_url.rstrip('/')
        self.embedding_model = embedding_model
        self.timeout = timeout
        if test_on_init:
            try:
                resp = self._call_api(["Test connection"])
                logger.info("Embedding API test OK. shape=%s", np.asarray(resp).shape)
            except Exception as e:
                logger.warning("Embedding API test failed: %s", e)
                raise

    def _parse_embedding_response(self, j):
        # 支援多種常見格式
        if isinstance(j, dict):
            if "embeddings" in j:
                return np.asarray(j["embeddings"], dtype=np.float32)
            if "embedding" in j:
                return np.asarray(j["embedding"], dtype=np.float32)
            if "data" in j and isinstance(j["data"], list):
                arr = []
                for it in j["data"]:
                    if isinstance(it, dict) and "embedding" in it:
                        arr.append(it["embedding"])
                if arr:
                    return np.asarray(arr, dtype=np.float32)
        if isinstance(j, list):
            return np.asarray(j, dtype=np.float32)
        raise ValueError("Unknown embedding response format")

    def _call_api(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        payload = {"text": texts}
        if self.embedding_model:
            payload["model"] = self.embedding_model  # 或 "embedding_model" 依你的 API 而定

        resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        j = resp.json()
        emb = self._parse_embedding_response(j)
        # 確保輸出為 2D（N, dim）
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

    def __call__(self, texts):
        return self._call_api(texts)

    def encode_documents(self, texts):
        return self.__call__(texts)

    def encode_queries(self, texts):
        return self.__call__(texts)

# ---- Vanna 主體：Milvus + vllm ----
class MyVanna(Milvus_VectorStore, Vllm):
    """Custom Vanna class integrating Milvus and vllm"""
    def __init__(self, config=None):
        Milvus_VectorStore.__init__(self, config=config)
        Vllm.__init__(self, config=config)


# 建立 Vanna 實例
def setup_vanna():
    milvus_client = MilvusClient(
        uri="http://10.13.18.40:19530",
        db_name="SQLRAG_ALS"   # 你的 Milvus 資料庫名
    )

    vn = MyVanna(config={
        'model_url': 'http://10.13.18.40:55700/v1',
        'model_name': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'milvus_client': milvus_client,
        'embedding_function': CustomEmbeddingFunction(api_url="http://10.13.18.40:14514/embed",
                                                     embedding_model="Conan-embedding-v1"),
        'n_results': 2,
    })
    # 這行可選，看你 Vanna 版本是否需要
    # vn.connect_to_database("SQLRAG_ALS")
    return vn

# 測試
if __name__ == "__main__":
    vn = setup_vanna()
    print(vn)  