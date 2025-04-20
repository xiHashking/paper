import os
from openai_client import OpenAIClient
from knowledge_graph import KnowledgeGraph
from rag_system import RAGSystem

def ensure_directory_exists(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def combined_answer(openai_client, question, rag_system, knowledge_graph):
    """
    结合RAG和知识图谱的问答方法
    
    参数:
        openai_client: OpenAI客户端实例
        question: 用户问题
        rag_system: RAG系统实例
        knowledge_graph: 知识图谱实例
        
    返回:
        回答文本
    """
    # 从RAG系统获取相关文档
    relevant_docs = rag_system.search(question, top_k=3)
    
    # 从知识图谱获取结构化知识
    kg_context = ""
    for u, v, data in knowledge_graph.graph.edges(data=True):
        relation = data.get('label', '与...相关')
        kg_context += f"{u} {relation} {v}。\n"
    
    # 构建组合上下文
    combined_context = "## 相关文档信息:\n\n"
    
    for idx, doc in enumerate(relevant_docs):
        source = doc["metadata"].get("source", "未知来源")
        combined_context += f"{idx + 1}. 来源({source}): {doc['text']}\n\n"
    
    combined_context += "\n## 知识图谱信息:\n\n" + kg_context
    
    # 使用OpenAI生成最终答案
    prompt = f"""
    同时基于以下两种信息来源回答问题：
    
    {combined_context}
    
    请同时利用文档的详细内容和知识图谱的结构化关系，给出最全面准确的回答。
    如果这些信息不足以回答问题，请说明无法提供完整回答。
    
    问题: {question}
    """
    
    response = openai_client.chat_completion([
        {"role": "system", "content": "你是一个高级智能助手，能够同时利用文档内容和知识图谱进行回答。"},
        {"role": "user", "content": prompt}
    ])
    
    return response

def main():
    """主程序"""
    # 确保必要的目录存在
    ensure_directory_exists("data")
    ensure_directory_exists("output")
    
    print("欢迎使用增强型OpenAI问答系统！")
    print("="*50)
    
    # 初始化OpenAI客户端
    try:
        openai_client = OpenAIClient()
        print("OpenAI客户端初始化成功！")
    except ValueError as e:
        print(f"错误：{e}")
        print("请在.env文件中设置正确的API密钥后重试。")
        return
    
    # 读取示例文本
    try:
        with open("data/sample_text.txt", "r", encoding="utf-8") as f:
            sample_text = f.read()
        print("已读取示例文本文件。")
    except FileNotFoundError:
        print("警告：找不到示例文本文件，将使用空文本。")
        sample_text = ""
    
    # 初始化并构建知识图谱
    print("\n构建知识图谱...")
    kg = KnowledgeGraph()
    if sample_text:
        # 使用spaCy方法构建知识图谱，不依赖OpenAI API
        kg.build_from_text(sample_text, method="spacy")
        kg.visualize(save_path="output/knowledge_graph.png")
        print("知识图谱已构建并保存到output/knowledge_graph.png")
        
        # 额外使用关键词方法构建知识图谱
        kg_keywords = KnowledgeGraph()
        kg_keywords.build_from_text(sample_text, method="keywords")
        kg_keywords.visualize(save_path="output/knowledge_graph_keywords.png")
        print("关键词知识图谱已构建并保存到output/knowledge_graph_keywords.png")
    else:
        print("无法构建知识图谱：没有文本数据。")
    
    # 初始化并构建RAG系统
    print("\n构建RAG系统...")
    try:
        rag = RAGSystem()
        
        if sample_text:
            rag.add_document(sample_text, metadata={"source": "示例文本"})
            # 构建索引
            rag.build_index()
            print("RAG系统已成功构建。")
        else:
            print("无法构建RAG系统：没有文本数据。")
    except Exception as e:
        print(f"RAG系统构建失败：{e}")
        print("请确保您已安装必要的依赖并配置了正确的模型路径。")
        rag = None
    
    # 交互式问答循环
    print("\n========== 开始交互式问答 ==========")
    print("您可以输入问题，系统将使用多种方法为您提供答案。")
    print("输入'exit'或'quit'退出。")
    
    while True:
        question = input("\n请输入您的问题：")
        if question.lower() in ["exit", "quit", "退出"]:
            break
        
        # 使用普通OpenAI回答
        try:
            print("\n方法1 - 直接使用OpenAI：")
            response = openai_client.chat_completion([
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": question}
            ])
            print(f"回答：{response}")
        except Exception as e:
            print(f"使用OpenAI回答失败：{e}")
        
        # 使用知识图谱回答
        print("\n方法2 - 基于知识图谱：")
        if sample_text:
            try:
                kg_response = kg.query_graph(question)
                print(f"回答：{kg_response}")
            except Exception as e:
                print(f"使用知识图谱回答失败：{e}")
        else:
            print("无法使用知识图谱回答：知识图谱未构建。")
        
        # 使用RAG系统回答
        print("\n方法3 - 基于RAG检索增强生成：")
        if rag and sample_text:
            try:
                # 移除model参数，使用默认model
                rag_response = rag.answer_question(question)
                print(f"回答：{rag_response}")
            except Exception as e:
                print(f"使用RAG系统回答失败：{e}")
        else:
            print("无法使用RAG系统回答：RAG系统未构建。")
        
        # 使用RAG和知识图谱结合回答
        print("\n方法4 - RAG与知识图谱结合：")
        if rag and sample_text:
            try:
                combined_response = combined_answer(openai_client, question, rag, kg)
                print(f"回答：{combined_response}")
            except Exception as e:
                print(f"使用RAG与知识图谱结合回答失败：{e}")
        else:
            print("无法使用RAG与知识图谱结合回答：系统未完全构建。")

if __name__ == "__main__":
    main() 