## 3.2 知识图谱构建

### 3.2.1 知识图谱设计

短篇小说知识图谱作为系统的关键组件，其设计直接影响对文学文本的结构化理解能力。本节详细说明知识图谱的设计思路和关键要素。

**1. 实体类型体系**

本研究根据短篇小说的特点，设计了以下核心实体类型体系：

- **人物实体（Character）**：小说中的角色，包括主角、配角和提及的其他人物
  - 属性：姓名、别称、身份、性别、年龄（如有）
  - 子类型：主角(Protagonist)、配角(Supporting)、背景角色(Background)

- **地点实体（Location）**：故事发生或提及的场所
  - 属性：名称、类型、描述
  - 子类型：室内(Indoor)、室外(Outdoor)、虚构(Fictional)、真实(Real)

- **事件实体（Event）**：故事中的关键事件和情节节点
  - 属性：名称、时间点、参与者、影响
  - 子类型：关键事件(KeyEvent)、背景事件(BackgroundEvent)

- **时间实体（Time）**：故事的时间背景和关键时间点
  - 属性：时间表达、相对位置（故事开始/中间/结束）
  - 子类型：具体时间(SpecificTime)、时间段(TimePeriod)、模糊时间(VagueTime)

- **主题元素（ThemeElement）**：象征、隐喻和主题相关的概念
  - 属性：名称、象征含义、出现位置
  - 子类型：象征物(Symbol)、主题概念(ThemeConcept)、情感元素(EmotionElement)

实体类型体系设计兼顾了通用性和针对性，既可适应不同类型的短篇小说，又能捕捉文学作品的特有元素。

**2. 关系类型设计**

为准确表达实体间的复杂关系，设计了多层次的关系类型体系：

- **人物关系（Character-Character）**
  - 社会关系：亲属(FamilyOf)、朋友(FriendOf)、同事(ColleagueOf)、对手(OpponentOf)等
  - 情感关系：爱(LovesFor)、恨(HatesFor)、敬(RespectsFor)、怕(FearsFor)等
  - 互动关系：帮助(Helps)、伤害(Hurts)、对话(TalksWith)、影响(Influences)等

- **人物-事件关系（Character-Event）**
  - 参与关系：参与(ParticipatesIn)、目睹(WitnessesIn)、引发(Triggers)、受影响(AffectedBy)等

- **人物-地点关系（Character-Location）**
  - 空间关系：位于(LocatesAt)、来自(ComesFrom)、前往(GoesTo)、居住(LivesIn)等

- **事件-时间关系（Event-Time）**
  - 时序关系：发生于(OccursAt)、先于(BeforeEvent)、后于(AfterEvent)、同时(SimultaneousWith)等

- **事件-地点关系（Event-Location）**
  - 场景关系：发生在(HappensAt)、影响(AffectsPlace)等

- **主题关联关系（Theme-Related）**
  - 象征关系：象征(Symbolizes)、暗示(Implies)、关联(AssociatesWith)等

这些关系类型使知识图谱能够捕捉到短篇小说中丰富的语义连接，支持对复杂叙事结构的表示。

**3. 属性设计**

除了核心的实体和关系，知识图谱还包含重要的属性信息：

- **实体属性**：
  - 出现位置(appearance_position)：实体在文本中首次出现的位置
  - 出现频率(frequency)：实体在文本中出现的次数
  - 重要性评分(importance_score)：基于中心度和频率计算的实体重要性

- **关系属性**：
  - 关系强度(strength)：表示关系的强弱程度，取值范围[0,1]
  - 关系极性(polarity)：表示关系的正负面性质，取值[-1,1]
  - 置信度(confidence)：表示关系提取的确信程度，取值[0,1]
  - 文本依据(text_evidence)：支持该关系的原文片段

这些属性信息丰富了知识图谱的表达能力，支持更精细的查询和推理。

**4. 图谱结构设计**

从技术实现角度，本研究采用了以下图谱结构设计：

- **图模型**：采用有向属性图(Directed Property Graph)模型
- **存储方式**：使用NetworkX库的DiGraph结构，支持节点和边的属性存储
- **索引机制**：建立基于实体名称和ID的索引，支持快速查找
- **元数据**：包含图谱构建时间、来源文本、构建方法等元信息

图谱设计注重平衡表达能力和实现复杂度，确保在捕捉丰富语义的同时保持查询效率和系统性能。

### 3.2.2 图谱构建方法实现

本研究实现了两种互补的知识图谱构建方法，并支持它们的混合使用：

**1. 基于spaCy的依存句法分析方法**

这种方法利用NLP工具的句法分析能力，提取结构明确的实体关系：

```python
def extract_triplets_with_spacy(self, text):
    """使用spaCy提取实体和关系"""
    doc = self.nlp(text)
    triplets = []
    
    # 基于依存关系的三元组提取
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
    
    return triplets
```

这种方法的核心步骤包括：

- 使用spaCy的依存句法分析器识别主语-谓语-宾语结构
- 识别命名实体并尝试找出它们之间的关系
- 基于语法规则提取可能的关系表达
- 分析共现实体之间的语义连接

基于依存句法的方法准确性较高，但可能会漏掉复杂表达的关系。

**2. 基于关键词的启发式方法**

为了弥补语法分析的局限性，实现了基于关键词的启发式关系提取方法：

```python
def extract_entities_with_keywords(self, text, keywords=None):
    """使用关键词和规则提取实体和关系"""
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
                            # 使用两个关键词之间的文本作为关系
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
```

这种方法的核心步骤包括：

- 提取文本中的关键词（名词、专有名词和命名实体）
- 分析同一句中出现的关键词对
- 提取关键词对之间的文本作为关系描述
- 对于相邻关键词，建立默认关联关系

基于关键词的方法覆盖率更高，能捕捉更多潜在关系，但可能包含更多噪声。

**3. 大型文本的分块处理机制**

为处理较长的短篇小说，实现了分块处理机制：

```python
def chunk_text(self, text, chunk_size=5000, overlap=500):
    """将大型文本分割成较小的块"""
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
```

分块处理的关键特点：

- 将长文本分割为可管理的小块
- 使用重叠区域保持上下文连贯性
- 在句子边界处分割，保持语义完整性
- 支持并行处理各个块，提高效率

**4. 图谱合并策略**

从多个文本块构建的子图谱需要合并为完整知识图谱：

```python
def merge_subgraphs(self, subgraphs):
    """合并多个子图为一个完整的知识图谱"""
    merged_graph = nx.DiGraph()
    
    # 合并节点
    for subgraph in subgraphs:
        for node, attr in subgraph.nodes(data=True):
            if node not in merged_graph:
                merged_graph.add_node(node, **attr)
            else:
                # 合并节点属性
                for key, value in attr.items():
                    if key in merged_graph.nodes[node]:
                        # 对于出现频率，累加计数
                        if key == 'frequency':
                            merged_graph.nodes[node][key] += value
                        # 对于出现位置，保留最早的位置
                        elif key == 'appearance_position' and value < merged_graph.nodes[node][key]:
                            merged_graph.nodes[node][key] = value
                        # 默认保留原有属性
                    else:
                        merged_graph.nodes[node][key] = value
    
    # 合并边
    for subgraph in subgraphs:
        for u, v, attr in subgraph.edges(data=True):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v, **attr)
            else:
                # 合并边属性
                for key, value in attr.items():
                    if key in merged_graph[u][v]:
                        # 对于关系强度，取最大值
                        if key == 'strength' and value > merged_graph[u][v][key]:
                            merged_graph[u][v][key] = value
                        # 对于置信度，取平均值
                        elif key == 'confidence':
                            merged_graph[u][v][key] = (merged_graph[u][v][key] + value) / 2
                        # 对于文本依据，合并不同来源
                        elif key == 'text_evidence':
                            if value not in merged_graph[u][v][key]:
                                merged_graph[u][v][key] += " | " + value
                    else:
                        merged_graph[u][v][key] = value
    
    return merged_graph
```

图谱合并的关键策略：

- 合并相同实体节点，整合其属性信息
- 保留最早的实体出现位置和最高的重要性分数
- 合并重复的关系边，整合关系属性
- 对于冲突的属性值，采用特定的解决策略（如取最大值、平均值或合并文本）

**5. 实体与关系的后处理优化**

为提高知识图谱质量，实现了一系列后处理优化步骤：

```python
def optimize_knowledge_graph(self):
    """对知识图谱进行后处理优化"""
    # 实体去重合并（处理别称和指代）
    self._merge_coreferent_entities()
    
    # 关系去噪（移除低置信度关系）
    edges_to_remove = []
    for u, v, attr in self.graph.edges(data=True):
        if attr.get('confidence', 1.0) < 0.3:  # 低于阈值的关系被移除
            edges_to_remove.append((u, v))
    for edge in edges_to_remove:
        self.graph.remove_edge(*edge)
    
    # 关系优化（合并同义关系）
    self._merge_similar_relations()
    
    # 计算实体重要性
    self._calculate_entity_importance()
    
    # 推断隐含关系
    self._infer_implicit_relations()
```

后处理优化包括：

- 实体统一：合并可能指向同一实体的不同表述
- 关系过滤：移除置信度低或支持证据不足的关系
- 关系规范化：将类似关系合并为标准化表达
- 实体重要性计算：基于网络中心度指标评估实体重要性
- 隐含关系推断：通过规则推理添加可能的隐含关系

通过这些后处理步骤，显著提高了知识图谱的质量和可用性。

### 3.2.3 知识图谱查询机制

构建完成的知识图谱需要高效的查询机制支持问答过程：

**1. 基于实体的查询**

实现了针对单个实体的多种查询方式：

```python
def query_entity_info(self, entity_name):
    """查询单个实体的详细信息"""
    if entity_name not in self.graph.nodes:
        return None
    
    # 获取实体属性
    entity_attrs = dict(self.graph.nodes[entity_name])
    
    # 获取实体关系
    relationships = []
    
    # 出边（该实体作为主体）
    for _, target, data in self.graph.out_edges(entity_name, data=True):
        relation = data.get('label', '与...相关')
        relationships.append({
            'direction': 'out',
            'relation': relation,
            'entity': target,
            'strength': data.get('strength', 1.0),
            'evidence': data.get('text_evidence', '')
        })
    
    # 入边（该实体作为客体）
    for source, _, data in self.graph.in_edges(entity_name, data=True):
        relation = data.get('label', '与...相关')
        relationships.append({
            'direction': 'in',
            'relation': relation,
            'entity': source,
            'strength': data.get('strength', 1.0),
            'evidence': data.get('text_evidence', '')
        })
    
    return {
        'entity': entity_name,
        'attributes': entity_attrs,
        'relationships': relationships
    }
```

**2. 关系路径查询**

实现了寻找两个实体之间关系路径的功能：

```python
def find_relationship_paths(self, entity1, entity2, max_length=3):
    """查找两个实体之间的所有关系路径"""
    if entity1 not in self.graph.nodes or entity2 not in self.graph.nodes:
        return []
    
    # 使用NetworkX的简单路径算法
    try:
        paths = list(nx.all_simple_paths(self.graph, entity1, entity2, cutoff=max_length))
    except nx.NetworkXNoPath:
        return []
    
    # 格式化路径结果
    result_paths = []
    for path in paths:
        path_with_relations = []
        for i in range(len(path)-1):
            source = path[i]
            target = path[i+1]
            relation = self.graph[source][target].get('label', '与...相关')
            evidence = self.graph[source][target].get('text_evidence', '')
            
            path_with_relations.append({
                'source': source,
                'relation': relation,
                'target': target,
                'evidence': evidence
            })
        
        result_paths.append(path_with_relations)
    
    return result_paths
```

**3. 高级模式查询**

实现了复杂的图谱模式匹配查询：

```python
def pattern_match_query(self, pattern):
    """基于模式的知识图谱查询
    
    pattern: 字典格式的查询模式，例如：
    {
        'start_entity': {'type': 'Character'},
        'relations': [
            {'type': 'FriendOf', 'direction': 'out'},
            {'type': 'ParticipatesIn', 'direction': 'out'}
        ],
        'end_entity': {'type': 'Event'}
    }
    """
    matches = []
    
    # 找到所有可能的起始实体
    start_entities = []
    if 'start_entity' in pattern:
        entity_type = pattern['start_entity'].get('type')
        for node, attrs in self.graph.nodes(data=True):
            if entity_type is None or attrs.get('type') == entity_type:
                start_entities.append(node)
    else:
        start_entities = list(self.graph.nodes())
    
    # 对每个起始实体执行路径匹配
    for start_entity in start_entities:
        if 'relations' not in pattern or not pattern['relations']:
            matches.append({'entities': [start_entity], 'relations': []})
            continue
        
        # 从起始实体开始路径匹配
        self._recursive_pattern_match(
            start_entity, 
            pattern['relations'], 
            0, 
            [start_entity], 
            [], 
            pattern.get('end_entity'), 
            matches
        )
    
    return matches
```

**4. 基于问题类型的查询路由**

实现了根据问题类型选择合适查询方法的路由机制：

```python
def query_graph(self, question):
    """基于问题进行知识图谱查询"""
    # 分析问题类型
    question_type = self._classify_question(question)
    
    # 提取问题中的关键实体
    entities = self._extract_question_entities(question)
    
    # 根据问题类型选择查询策略
    if question_type == "entity_info" and entities:
        # 实体信息查询
        return self.query_entity_info(entities[0])
        
    elif question_type == "relationship" and len(entities) >= 2:
        # 关系查询
        return self.find_relationship_paths(entities[0], entities[1])
        
    elif question_type == "pattern":
        # 构建查询模式
        pattern = self._build_query_pattern(question)
        return self.pattern_match_query(pattern)
        
    elif question_type == "general":
        # 一般问题，使用综合查询策略
        return self._comprehensive_query(question, entities)
    
    # 如果无法分类或提取实体，返回空结果
    return None
```

这些查询机制使知识图谱能够有效支持不同类型的问题，为混合问答系统提供结构化知识基础。

### 3.2.4 知识图谱可视化

为了直观展示短篇小说的结构化理解，实现了知识图谱可视化功能：

```python
def visualize(self, figsize=(12, 10), save_path=None):
    """可视化知识图谱"""
    plt.figure(figsize=figsize)
    
    # 设置布局算法
    pos = nx.spring_layout(self.graph, k=0.15, iterations=50)
    
    # 绘制节点
    nx.draw_networkx_nodes(
        self.graph, pos, 
        node_size=[self.graph.nodes[n].get('importance_score', 1) * 300 for n in self.graph.nodes],
        node_color='skyblue', 
        alpha=0.8
    )
    
    # 绘制边
    nx.draw_networkx_edges(
        self.graph, pos, 
        width=1.0, 
        alpha=0.5, 
        edge_color='gray',
        arrows=True, 
        arrowstyle='-|>', 
        arrowsize=10
    )
    
    # 绘制标签
    nx.draw_networkx_labels(
        self.graph, pos, 
        font_size=10, 
        font_family='SimHei',  # 使用中文兼容字体
        font_weight='bold'
    )
    
    # 绘制边标签（关系）
    edge_labels = {(u, v): d.get('label', '') for u, v, d in self.graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        self.graph, pos, 
        edge_labels=edge_labels, 
        font_size=8,
        font_family='SimHei'
    )
    
    plt.title("短篇小说知识图谱", fontsize=15, fontfamily='SimHei')
    plt.axis('off')  # 关闭坐标轴
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
        print(f"知识图谱已保存至: {save_path}")
    
    plt.show()
```

可视化功能的关键特点：

- 节点大小根据实体重要性动态调整
- 使用不同颜色区分实体类型
- 显示关系标签，直观反映实体间关系
- 采用弹簧布局算法，优化图的可读性
- 支持中文显示和高分辨率图像导出

通过可视化，研究者和用户可以直观把握小说的结构特征和关键元素，辅助文学分析和理解。

知识图谱构建模块通过上述设计和实现，为短篇小说提供了结构化的语义表示，捕捉了人物关系、情节发展和主题元素，为后续的混合问答提供了重要支持。 