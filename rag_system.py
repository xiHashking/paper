import os
import numpy as np
import faiss
from openai_client import OpenAIClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGSystem:
    def __init__(self):
        """初始化RAG系统"""
        self.openai_client = OpenAIClient()
        self.documents = []
        self.embeddings = []
        self.index = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def add_document(self, text, metadata=None):
        """
        添加文档到RAG系统
        
        参数:
            text (str): 文档文本
            metadata (dict): 文档元数据
        """
        chunks = self.text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_id"] = i
            
            self.documents.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
    
    def add_documents_from_directory(self, directory, extensions=[".txt"]):
        """
        从目录添加文档
        
        参数:
            directory (str): 目录路径
            extensions (list): 支持的文件扩展名
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    self.add_document(content, metadata={"source": file_path})
    
    def build_index(self):
        """构建向量索引"""
        if not self.documents:
            print("没有文档，无法构建索引")
            return
        
        # 为每个文档生成嵌入向量
        self.embeddings = []
        for doc in self.documents:
            embedding = self.openai_client.embeddings(doc["text"])
            if embedding:
                self.embeddings.append(embedding)
            else:
                print(f"警告: 无法为文档生成嵌入向量: {doc['text'][:50]}...")
        
        if not self.embeddings:
            print("没有可用的嵌入向量，无法构建索引")
            return
        
        # 创建FAISS索引
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))
        print(f"索引已构建，包含 {len(self.embeddings)} 个文档嵌入向量")
    
    def search(self, query, top_k=3):
        """
        搜索相关文档
        
        参数:
            query (str): 查询文本
            top_k (int): 返回的最相关文档数量
            
        返回:
            list: 相关文档列表
        """
        if not self.index:
            print("索引尚未构建，请先调用build_index()")
            return []
        
        # 获取查询的嵌入向量
        query_embedding = self.openai_client.embeddings(query)
        if not query_embedding:
            print("无法为查询生成嵌入向量")
            return []
        
        # 搜索最相似的文档
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            min(top_k, len(self.embeddings))
        )
        
        # 返回相关文档
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def answer_question(self, question, top_k=3):
        """
        回答问题
        
        参数:
            question (str): 问题
            top_k (int): 使用的相关文档数量
            
        返回:
            str: 回答
        """
        # 搜索相关文档
        relevant_docs = self.search(question, top_k=top_k)
        
        if not relevant_docs:
            return "无法找到相关信息来回答您的问题。"
        
        # 构建上下文
        context = ""
        for doc in relevant_docs:
            source = doc["metadata"].get("source", "未知来源")
            context += f"文档（来源: {source}）: {doc['text']}\n\n"
        
        # 使用OpenAI生成回答
        prompt = f"""
        基于以下文档信息，回答问题。如果文档中没有足够的信息来回答问题，请说明无法回答。

        文档:
        {context}

        问题: {question}
        """
        
        response = self.openai_client.chat_completion([
            {"role": "system", "content": "你是一个基于检索的问答助手。请基于提供的文档内容回答问题，不要使用你自己的知识。"},
            {"role": "user", "content": prompt}
        ])
        
        return response

# 测试代码
if __name__ == "__main__":
    rag = RAGSystem()
    
    # 添加示例文档
    rag.add_documents_from_directory("data")
    
    # 构建索引
    rag.build_index()
    
    # 测试问答
    question = "深度学习与机器学习的关系是什么？"
    answer = rag.answer_question(question)
    print(f"问题: {question}")
    print(f"回答: {answer}")