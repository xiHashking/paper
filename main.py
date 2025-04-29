import os
from openai_client import OpenAIClient
from knowledge_graph import KnowledgeGraph
from rag_system import RAGSystem
import re
import json
from collections import Counter

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

# 改进后的评估器类
class ImprovedResponseEvaluator:
    """使用更合理标准评估回答质量的工具"""
    
    def __init__(self, knowledge_base: str):
        """初始化评估器"""
        # 将知识库文本拆分为句子列表
        self.knowledge_base = re.split(r'[。！？]', knowledge_base)
        self.knowledge_base = [s.strip() for s in self.knowledge_base if s.strip()]
        
        # 提取知识库中的关键词和实体
        self.kb_words, self.kb_entities = self._extract_keywords_and_entities(knowledge_base)
        
        # 识别知识库中的概念定义句
        self.definitions = self._identify_definitions(self.knowledge_base)
    
    def _extract_keywords_and_entities(self, text):
        """提取文本中的关键词和可能的实体"""
        # 移除标点符号和常见停用词
        cleaned_text = re.sub(r'[，。！？、；：""''（）【】\s]', ' ', text)
        words = cleaned_text.split()
        
        # 简单的停用词列表
        stopwords = ['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
        
        # 过滤掉停用词
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        
        # 尝试识别实体（大写字母开头或包含括号的词）
        entities = set()
        for sentence in self.knowledge_base:
            # 查找形如 "XXX是..." 或 "XXX（YYY）是..." 的模式
            entity_matches = re.findall(r'([^，。！？、；：]*?)\s*(?:\([^)]+\))?\s*是', sentence)
            for match in entity_matches:
                if match.strip() and len(match.strip()) > 1:
                    entities.add(match.strip())
            
            # 查找圆括号中的内容，可能是缩写或专有名词
            abbr_matches = re.findall(r'\(([^)]+)\)', sentence)
            for match in abbr_matches:
                if match.strip() and len(match.strip()) > 1:
                    entities.add(match.strip())
        
        return Counter(keywords), entities
    
    def _identify_definitions(self, sentences):
        """识别知识库中的概念定义句"""
        definitions = {}
        
        for sentence in sentences:
            # 匹配形如"X是Y"或"X(A)是Y"的定义句
            matches = re.findall(r'([^\s，。！？:；（）]+(?:\([^)]+\))?)\s*是\s*([^。！？]+)', sentence)
            for match in matches:
                if match[0].strip() and match[1].strip():
                    definitions[match[0].strip()] = match[1].strip()
        
        return definitions
    
    def content_completeness(self, response: str) -> float:
        """评估回答的内容完整性（基于句子数量和长度）"""
        # 将回答拆分成句子
        sentences = re.split(r'[。！？]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 计算平均句子长度
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 根据句子数量和平均长度计算完整性得分
        sentence_count_score = min(len(sentences) / 3, 1.0)  # 3句或以上得满分
        length_score = min(avg_length / 15, 1.0)  # 平均15字或以上得满分
        
        # 组合两个分数
        return (sentence_count_score * 0.6 + length_score * 0.4)
    
    def entity_coverage(self, response: str) -> float:
        """评估回答中实体的覆盖率"""
        # 计算回答中包含了多少知识库中的实体
        mentioned_entities = 0
        
        for entity in self.kb_entities:
            if entity in response:
                mentioned_entities += 1
        
        if not self.kb_entities:
            return 0.5  # 如果没有识别到实体，给一个中等分数
            
        return min(mentioned_entities / len(self.kb_entities) * 3, 1.0)  # 覆盖1/3的实体就得满分
    
    def definition_accuracy(self, response: str) -> float:
        """评估回答中定义的准确性"""
        if not self.definitions:
            return 0.5  # 如果没有定义基准，给一个中等分数
        
        # 计算回答中的定义与知识库中的定义的匹配程度
        # 查找回答中可能的定义句
        potential_definitions = re.findall(r'([^\s，。！？:；（）]+(?:\([^)]+\))?)\s*是\s*([^。！？]+)', response)
        
        if not potential_definitions:
            return 0.0  # 没有找到定义
        
        correct_defs = 0
        
        for term, definition in potential_definitions:
            term = term.strip()
            if term in self.definitions:
                # 计算定义的相似度（简单版本：共同词的比例）
                kb_def_words = set(self.definitions[term].split())
                response_def_words = set(definition.split())
                
                if not kb_def_words or not response_def_words:
                    continue
                    
                overlap = len(kb_def_words.intersection(response_def_words))
                similarity = overlap / max(len(kb_def_words), len(response_def_words))
                
                if similarity >= 0.3:  # 如果定义有30%的词重叠，视为正确
                    correct_defs += 1
        
        return min(correct_defs / len(self.definitions) * 2, 1.0)  # 正确定义半数概念就得满分
    
    def knowledge_base_alignment(self, response: str) -> float:
        """评估回答与知识库的整体一致性"""
        # 提取回答中的关键词
        response_words, _ = self._extract_keywords_and_entities(response)
        response_words_set = set(response_words.keys())
        
        # 知识库关键词
        kb_words_set = set(self.kb_words.keys())
        
        if not kb_words_set or not response_words_set:
            return 0.0
            
        # 计算关键词重叠
        overlap = kb_words_set.intersection(response_words_set)
        alignment_score = len(overlap) / min(len(kb_words_set), len(response_words_set) * 2)
        
        return min(alignment_score * 1.5, 1.0)  # 放宽标准，覆盖2/3的关键词就得满分
    
    def evaluate(self, response: str) -> dict:
        """综合评估回答质量"""
        completeness = self.content_completeness(response)
        entity_cover = self.entity_coverage(response)
        definition_acc = self.definition_accuracy(response)
        kb_alignment = self.knowledge_base_alignment(response)
        
        # 计算加权得分，调整权重以优先考虑内容完整性和知识覆盖
        weights = {
            "内容完整性": 0.35,  # 增加完整性的权重
            "实体覆盖率": 0.25,
            "定义准确性": 0.2,
            "知识库一致性": 0.2
        }
        
        overall_score = (
            weights["内容完整性"] * completeness +
            weights["实体覆盖率"] * entity_cover +
            weights["定义准确性"] * definition_acc +
            weights["知识库一致性"] * kb_alignment
        )
        
        return {
            "内容完整性": completeness,
            "实体覆盖率": entity_cover,
            "定义准确性": definition_acc,
            "知识库一致性": kb_alignment,
            "综合得分": overall_score
        }
    
    def compare_systems(self, rag_only_response: str, rag_kg_response: str) -> dict:
        """比较仅RAG和RAG+KG系统的回答质量"""
        results = {
            "仅RAG": self.evaluate(rag_only_response),
            "RAG+知识图谱": self.evaluate(rag_kg_response)
        }
        
        # 按维度比较
        dimension_comparison = {}
        for dimension in results["仅RAG"].keys():
            if dimension != "综合得分":
                rag_score = results["仅RAG"][dimension]
                rag_kg_score = results["RAG+知识图谱"][dimension]
                
                if rag_score > 0:
                    improvement = ((rag_kg_score / rag_score) - 1) * 100
                elif rag_kg_score > 0:
                    improvement = float("inf")
                else:
                    improvement = 0.0
                    
                dimension_comparison[dimension] = {
                    "仅RAG": rag_score,
                    "RAG+知识图谱": rag_kg_score,
                    "改进百分比": improvement
                }
        
        # 综合评分比较
        if results["仅RAG"]["综合得分"] > 0:
            overall_improvement = ((results["RAG+知识图谱"]["综合得分"] / results["仅RAG"]["综合得分"]) - 1) * 100
        elif results["RAG+知识图谱"]["综合得分"] > 0:
            overall_improvement = float("inf")
        else:
            overall_improvement = 0.0
        
        # 确定更好的系统
        better_system = "RAG+知识图谱" if results["RAG+知识图谱"]["综合得分"] > results["仅RAG"]["综合得分"] else "仅RAG"
        
        return {
            "详细评分": results,
            "各维度比较": dimension_comparison,
            "综合改进百分比": overall_improvement,
            "更好的系统": better_system
        }

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
    
    # 初始化评估器（使用改进的评估器）
    if sample_text:
        evaluator = ImprovedResponseEvaluator(sample_text)
        print("\n评估系统已初始化。")
    else:
        evaluator = None
        print("\n无法初始化评估系统：没有文本数据。")
    
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
        rag_response = None
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
        combined_response = None
        print("\n方法4 - RAG与知识图谱结合：")
        if rag and sample_text:
            try:
                combined_response = combined_answer(openai_client, question, rag, kg)
                print(f"回答：{combined_response}")
            except Exception as e:
                print(f"使用RAG与知识图谱结合回答失败：{e}")
        else:
            print("无法使用RAG与知识图谱结合回答：系统未完全构建。")
        
        # 评估方法3和方法4的回答
        if evaluator and rag_response and combined_response:
            try:
                print("\n=========== 质量评估结果 ===========")
                evaluation_results = evaluator.compare_systems(rag_response, combined_response)
                print(json.dumps(evaluation_results, ensure_ascii=False, indent=2))
                
                # 添加明确的结论
                better_system = evaluation_results["更好的系统"]
                improvement = evaluation_results["综合改进百分比"]
                
                print("\n=== 评估结论 ===")
                if better_system == "RAG+知识图谱":
                    print(f"✓ RAG+知识图谱方法效果更好，提升了 {improvement:.2f}%")
                else:
                    print(f"✓ 仅RAG方法效果更好，RAG+知识图谱表现较差 {-improvement:.2f}%")
                print("="*40)
            except Exception as e:
                print(f"评估回答失败：{e}")
                print(f"错误详情: {str(e)}")

if __name__ == "__main__":
    main() 