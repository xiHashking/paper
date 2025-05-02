import os
import numpy as np
import faiss
from openai_client import OpenAIClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import time
import pickle

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
    
    def chunk_documents(self, batch_size=100):
        """
        将文档分成批次处理
        
        参数:
            batch_size (int): 每批处理的文档数量
            
        返回:
            list: 文档批次列表
        """
        batches = []
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:min(i + batch_size, len(self.documents))]
            batches.append(batch)
        return batches
    
    def build_index_in_batches(self, batch_size=100, temp_dir=None):
        """
        分批构建向量索引，适用于大型文档集合
        
        参数:
            batch_size (int): 每批处理的文档数量
            temp_dir (str): 临时文件保存目录，如果为None则不保存临时文件
        """
        if not self.documents:
            print("没有文档，无法构建索引")
            return
        
        # 创建临时目录
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            print(f"临时文件将保存在: {temp_dir}")
        
        # 分批获取文档
        batches = self.chunk_documents(batch_size)
        print(f"文档已分成{len(batches)}个批次进行处理，每批{batch_size}个文档")
        
        # 跟踪文档索引映射
        doc_indices = {}
        all_embeddings = []
        
        # 逐批处理文档
        for batch_idx, batch in enumerate(tqdm(batches, desc="处理文档批次")):
            print(f"\n处理第{batch_idx+1}/{len(batches)}批文档...")
            batch_embeddings = []
            
            # 处理批次中的每个文档
            for doc_idx, doc in enumerate(tqdm(batch, desc=f"批次{batch_idx+1}中的文档")):
                global_idx = batch_idx * batch_size + doc_idx
                doc_indices[global_idx] = doc
                
                # 生成嵌入向量
                embedding = self.openai_client.embeddings(doc["text"])
                if embedding:
                    batch_embeddings.append(embedding)
                    all_embeddings.append(embedding)
                else:
                    print(f"警告: 无法为文档生成嵌入向量: {doc['text'][:30]}...")
                
                # 避免API限制
                time.sleep(0.1)
            
            # 保存批次结果
            if temp_dir:
                # 保存批次嵌入向量
                with open(os.path.join(temp_dir, f"embeddings_batch_{batch_idx}.pkl"), "wb") as f:
                    pickle.dump(batch_embeddings, f)
                
                # 保存批次文档索引
                batch_indices = {global_idx: doc for global_idx, doc in doc_indices.items() 
                               if batch_idx * batch_size <= global_idx < (batch_idx + 1) * batch_size}
                with open(os.path.join(temp_dir, f"docs_batch_{batch_idx}.pkl"), "wb") as f:
                    pickle.dump(batch_indices, f)
        
        # 存储最终的嵌入向量
        self.embeddings = all_embeddings
        
        # 如果没有嵌入向量，返回
        if not self.embeddings:
            print("没有可用的嵌入向量，无法构建索引")
            return
        
        # 创建并构建FAISS索引
        print("构建FAISS索引...")
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))
        
        # 保存最终索引
        if temp_dir:
            faiss.write_index(self.index, os.path.join(temp_dir, "faiss_index.bin"))
            with open(os.path.join(temp_dir, "document_map.pkl"), "wb") as f:
                pickle.dump(doc_indices, f)
        
        print(f"索引已构建，包含 {len(self.embeddings)} 个文档嵌入向量")
    
    def load_index_from_files(self, index_path, doc_map_path):
        """
        从文件加载索引和文档映射
        
        参数:
            index_path (str): FAISS索引文件路径
            doc_map_path (str): 文档映射文件路径
        """
        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        print(f"已加载FAISS索引，包含 {self.index.ntotal} 个向量")
        
        # 加载文档映射
        with open(doc_map_path, "rb") as f:
            doc_map = pickle.load(f)
        
        # 重建文档列表
        self.documents = [doc for _, doc in sorted(doc_map.items())]
        
        # 初始化embeddings数组，确保长度与索引中的向量数量一致
        # 实际的嵌入向量存储在FAISS索引中
        self.embeddings = [None] * self.index.ntotal
        
        print(f"已加载 {len(self.documents)} 个文档")
    
    def build_index(self, batch_mode=False, batch_size=100, temp_dir=None):
        """构建向量索引，支持批处理模式"""
        if batch_mode:
            self.build_index_in_batches(batch_size, temp_dir)
        else:
            # 原有的build_index方法
            if not self.documents:
                print("没有文档，无法构建索引")
                return
            
            # 为每个文档生成嵌入向量
            self.embeddings = []
            for doc in tqdm(self.documents, desc="生成嵌入向量"):
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
        
        # 确保top_k是有效的
        if top_k <= 0:
            print("警告: top_k必须大于0，使用默认值3")
            top_k = 3
        
        # 计算可搜索的文档数量 - 使用索引中的向量数量
        max_docs = min(top_k, self.index.ntotal)
        
        # 搜索最相似的文档
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            max_docs
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
    
    # 使用批处理模式构建索引
    rag.build_index(batch_mode=True, batch_size=50, temp_dir="temp_embeddings")
    
    # 测试问答
    question = "深度学习与机器学习的关系是什么？"
    answer = rag.answer_question(question)
    print(f"问题: {question}")
    print(f"回答: {answer}")