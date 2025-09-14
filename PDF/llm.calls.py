from vanna.vllm import Vllm
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

# 繼承 VectorStore + Vllm
class MyVanna(ChromaDB_VectorStore, Vllm):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Vllm.__init__(self, config=config)


# 這裡把 API 換成本地 Vllm
vn = MyVanna(config={
    "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",            
    "host": "http://10.13.18.40:55700/v1"
    "api_key": "not-used"
})

sql = vn.generate_sql("列出所有使用者的姓名")
print(sql)