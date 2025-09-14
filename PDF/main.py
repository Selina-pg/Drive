from vanna.ollama import Ollama
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

# 繼承 VectorStore + Ollama
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


# 這裡把 API 換成本地 LLaMA (Ollama)
vn = MyVanna(config={
    "model": "llama3.1:8b",            
    "ollama_host": "http://localhost:11434"
})

sql = vn.generate_sql("列出所有使用者的姓名")
print(sql)