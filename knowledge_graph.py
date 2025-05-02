import spacy
import networkx as nx
import matplotlib.pyplot as plt
from openai_client import OpenAIClient
import matplotlib
import os
import time
from tqdm import tqdm
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体

class KnowledgeGraph:
    def __init__(self):
        """初始化知识图谱构建器"""
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            print("下载中文语言模型...")
            spacy.cli.download("zh_core_web_sm")
            self.nlp = spacy.load("zh_core_web_sm")
        self.graph = nx.DiGraph()
        self.openai_client = OpenAIClient()  # 仅用于最终问答
    
    def extract_triplets_with_spacy(self, text):
        """
        使用spaCy提取实体和关系
        """
        doc = self.nlp(text)
        triplets = []
        
        # 简单的基于依存关系的三元组提取
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "attr", "prep"):
                            obj = child.text
                            # 对于介词短语，获取整个短语
                            if child.dep_ == "prep":
                                for prep_child in child.children:
                                    obj += " " + prep_child.text
                            triplets.append((subject, verb, obj))
        
        # 基于实体识别的简单关系提取
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        for i, (entity1, label1) in enumerate(entities):
            for j, (entity2, label2) in enumerate(entities):
                if i != j and entity1 in text and entity2 in text:
                    # 检查两个实体是否在同一句话中
                    for sent in doc.sents:
                        sent_text = sent.text
                        if entity1 in sent_text and entity2 in sent_text:
                            # 寻找连接实体的动词或短语
                            # 这只是一个简化版本，实际应用中可能需要更复杂的逻辑
                            start_idx = min(sent_text.find(entity1), sent_text.find(entity2))
                            end_idx = max(sent_text.find(entity1) + len(entity1), 
                                        sent_text.find(entity2) + len(entity2))
                            # 只考虑实体之间的文本
                            between_text = sent_text[start_idx:end_idx]
                            
                            # 使用spaCy分析中间文本
                            between_doc = self.nlp(between_text)
                            verbs = [token.text for token in between_doc if token.pos_ == "VERB"]
                            
                            if verbs:
                                relation = verbs[0]  # 简单地使用第一个动词作为关系
                                triplets.append((entity1, relation, entity2))
        
        # 如果没有提取到三元组，尝试更简单的方法
        if not triplets:
            for ent1 in doc.ents:
                for ent2 in doc.ents:
                    if ent1.text != ent2.text:
                        # 简单地用"相关"作为关系
                        triplets.append((ent1.text, "相关", ent2.text))
        
        return triplets
    
    def extract_entities_with_keywords(self, text, keywords=None):
        """
        使用关键词和规则提取实体和关系
        
        参数:
            text (str): 文本内容
            keywords (list): 关键词列表，如果为None则自动提取
        """
        doc = self.nlp(text)
        triplets = []
        
        # 如果没有提供关键词，则提取名词和命名实体作为关键词
        if keywords is None:
            # 提取所有名词和命名实体
            nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
            entities = [ent.text for ent in doc.ents]
            keywords = list(set(nouns + entities))
        
        # 在每个句子中查找关键词对之间的关系
        for sent in doc.sents:
            sent_text = sent.text
            found_keywords = [kw for kw in keywords if kw in sent_text]
            
            # 如果句子中至少有两个关键词，则提取关系
            if len(found_keywords) >= 2:
                for i, kw1 in enumerate(found_keywords):
                    for kw2 in found_keywords[i+1:]:
                        # 获取两个关键词之间的文本
                        idx1 = sent_text.find(kw1)
                        idx2 = sent_text.find(kw2)
                        
                        if idx1 < idx2:
                            between = sent_text[idx1 + len(kw1):idx2].strip()
                            if between:
                                # 简单地使用两个关键词之间的文本作为关系
                                triplets.append((kw1, between, kw2))
                            else:
                                # 如果两个关键词之间没有文本，使用"相关"作为关系
                                triplets.append((kw1, "相关", kw2))
                        else:
                            between = sent_text[idx2 + len(kw2):idx1].strip()
                            if between:
                                triplets.append((kw2, between, kw1))
                            else:
                                triplets.append((kw2, "相关", kw1))
        
        return triplets
    
    def chunk_text(self, text, chunk_size=5000, overlap=500):
        """
        将大型文本分割成较小的块
        
        参数:
            text (str): 输入文本
            chunk_size (int): 每个块的最大字符数
            overlap (int): 块之间的重叠字符数，用于保持上下文连贯性
            
        返回:
            list: 文本块列表
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # 如果不是最后一块，尝试找到句子边界来分割
            if end < text_length:
                # 在重叠区域内寻找句号、问号或感叹号作为分割点
                boundary = end
                search_end = min(end + overlap, text_length)
                for i in range(end, search_end):
                    if text[i] in ["。", "！", "？", ".", "!", "?"]:
                        boundary = i + 1
                        break
                
                chunk = text[start:boundary]
                start = boundary - overlap  # 减去重叠部分，保持上下文连贯性
            else:
                # 最后一块
                chunk = text[start:end]
                start = end
            
            if chunk.strip():  # 确保块不是空的
                chunks.append(chunk)
        
        return chunks
    
    def build_from_text(self, text, method="spacy", keywords=None):
        """
        从文本构建知识图谱
        
        参数:
            text (str): 输入文本
            method (str): 提取方法，可选 "spacy" 或 "keywords"
            keywords (list): 关键词列表，仅在method="keywords"时使用
        """
        if method == "spacy":
            triplets = self.extract_triplets_with_spacy(text)
        elif method == "keywords":
            triplets = self.extract_entities_with_keywords(text, keywords)
        else:
            raise ValueError("不支持的方法。请使用 'spacy' 或 'keywords'。")
        
        for subject, relation, obj in triplets:
            self.graph.add_node(subject)
            self.graph.add_node(obj)
            self.graph.add_edge(subject, obj, label=relation)
        
        return self.graph
    
    def build_from_large_text(self, text, method="spacy", chunk_size=5000, overlap=500, 
                              keywords=None, temp_save_path=None, merge_subgraphs=True):
        """
        从大型文本构建知识图谱，使用分块处理方法
        
        参数:
            text (str): 输入文本
            method (str): 提取方法，可选 "spacy" 或 "keywords"
            chunk_size (int): 每个文本块的大小
            overlap (int): 块之间的重叠字符数
            keywords (list): 关键词列表，仅在method="keywords"时使用
            temp_save_path (str): 临时保存路径，如果提供则每个块处理后保存子图
            merge_subgraphs (bool): 是否合并所有子图，False则只保留最后一个子图
            
        返回:
            nx.DiGraph: 构建的知识图谱
        """
        # 清空现有图谱
        self.graph = nx.DiGraph()
        
        # 分割文本
        chunks = self.chunk_text(text, chunk_size, overlap)
        print(f"文本已分割为{len(chunks)}个块进行处理")
        
        # 存储所有子图
        subgraphs = []
        
        # 处理每个文本块
        for i, chunk in enumerate(tqdm(chunks, desc="处理文本块")):
            # 创建一个临时图谱
            temp_graph = nx.DiGraph()
            
            # 从当前块提取三元组
            if method == "spacy":
                triplets = self.extract_triplets_with_spacy(chunk)
            elif method == "keywords":
                triplets = self.extract_entities_with_keywords(chunk, keywords)
            else:
                raise ValueError("不支持的方法。请使用 'spacy' 或 'keywords'。")
            
            # 将三元组添加到临时图谱
            for subject, relation, obj in triplets:
                temp_graph.add_node(subject)
                temp_graph.add_node(obj)
                temp_graph.add_edge(subject, obj, label=relation)
            
            # 保存子图
            subgraphs.append(temp_graph)
            
            # 如果提供了临时保存路径，保存当前子图
            if temp_save_path:
                os.makedirs(temp_save_path, exist_ok=True)
                nx.write_gml(temp_graph, os.path.join(temp_save_path, f"subgraph_{i}.gml"))
            
            # 合并到主图或替换主图
            if merge_subgraphs:
                self.graph = nx.compose(self.graph, temp_graph)
            else:
                self.graph = temp_graph
            
            # 可选：给处理器一点休息时间，避免过度占用CPU
            time.sleep(0.01)
        
        print(f"知识图谱构建完成，共有{self.graph.number_of_nodes()}个节点和{self.graph.number_of_edges()}条边")
        return self.graph
    
    def load_from_subgraphs(self, subgraph_dir, merge_all=True):
        """
        从保存的子图文件加载知识图谱
        
        参数:
            subgraph_dir (str): 子图文件目录
            merge_all (bool): 是否合并所有子图，False则按最新的子图文件加载
            
        返回:
            nx.DiGraph: 加载的知识图谱
        """
        # 清空现有图谱
        self.graph = nx.DiGraph()
        
        # 获取目录中的所有GML文件
        subgraph_files = [f for f in os.listdir(subgraph_dir) if f.endswith('.gml')]
        
        if not subgraph_files:
            print(f"警告：在{subgraph_dir}中没有找到子图文件")
            return self.graph
        
        # 按文件名排序（假设格式为 subgraph_{i}.gml）
        subgraph_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        
        print(f"找到{len(subgraph_files)}个子图文件")
        
        if merge_all:
            # 加载并合并所有子图
            for file_name in tqdm(subgraph_files, desc="加载子图"):
                file_path = os.path.join(subgraph_dir, file_name)
                try:
                    subgraph = nx.read_gml(file_path)
                    self.graph = nx.compose(self.graph, subgraph)
                    #print(f"已加载并合并{file_name}")
                except Exception as e:
                    print(f"加载{file_name}失败: {e}")
        else:
            # 只加载最新的子图
            latest_file = subgraph_files[-1]
            file_path = os.path.join(subgraph_dir, latest_file)
            try:
                self.graph = nx.read_gml(file_path)
                print(f"已加载最新子图{latest_file}")
            except Exception as e:
                print(f"加载{latest_file}失败: {e}")
        
        print(f"知识图谱加载完成，共有{self.graph.number_of_nodes()}个节点和{self.graph.number_of_edges()}条边")
        return self.graph
    
    def visualize(self, figsize=(12, 10), save_path=None):
        """
        可视化知识图谱
        
        参数:
            figsize (tuple): 图像大小
            save_path (str): 保存路径，如果为None则显示图像
        """
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, node_size=3000, node_color="lightblue", alpha=0.8)
        nx.draw_networkx_labels(self.graph, pos, font_size=12)
        
        # 绘制边和边标签
        nx.draw_networkx_edges(self.graph, pos, width=2, alpha=0.5, arrows=True)
        edge_labels = {(u, v): d["label"] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)
        
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"图形已保存到 {save_path}")
        else:
            plt.show()
    
    def query_graph(self, query_text):
        """
        查询知识图谱
        
        参数:
            query_text (str): 查询文本
        
        返回:
            str: 查询结果
        """
        # 使用OpenAI生成查询结果
        context = ""
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('label', '与...相关')
            context += f"{u} {relation} {v}。\n"
        
        prompt = f"""
        基于以下知识图谱信息：
        
        {context}
        
        回答问题：{query_text}
        """
        
        response = self.openai_client.chat_completion([
            {"role": "system", "content": "你是一个基于知识图谱进行问答的助手。"},
            {"role": "user", "content": prompt}
        ])
        
        return response

# 测试代码
if __name__ == "__main__":
    with open("data/sample_text.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    kg = KnowledgeGraph()
    
    # 测试标准方法
    # kg.build_from_text(text, method="spacy")
    
    # 测试分块处理方法
    kg.build_from_large_text(text, method="spacy", chunk_size=1000, overlap=200, 
                            temp_save_path="temp_graphs")
    kg.visualize(save_path="output/knowledge_graph.png")
    
    # 示例查询
    query = "深度学习与机器学习有什么关系？"
    result = kg.query_graph(query)
    print(f"查询: {query}")
    print(f"回答: {result}") 