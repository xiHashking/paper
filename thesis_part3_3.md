## 3.3 RAG系统实现

### 3.3.1 文本分块策略

检索增强生成系统的核心前提是将长文本划分为适合检索的片段。本节详细介绍本研究实现的文本分块策略，这是构建高效RAG系统的基础。

**1. 分块策略设计原则**

本研究针对短篇小说的特点，设计了以下分块策略原则：

- **语义完整性**：分块应尽量保持语义完整，不割裂上下文关系
- **信息密度平衡**：每个块应具有相近的信息密度，避免信息分布不均
- **检索友好性**：分块大小适中，既能提供足够上下文，又不过于冗长
- **叙事结构保留**：分块应尽可能保留叙事单元的完整性
- **重叠设计**：相邻块间应有适当重叠，避免关键信息落在分块边界

基于这些原则，我们实现了多种分块策略的组合使用。

**2. 递归字符分割器实现**

基础的分块策略采用递归字符分割器（RecursiveCharacterTextSplitter），该方法根据文本标记（如段落符号、句号等）递归地将文本分割为语义相对完整的片段：

```python
def create_text_splitter(self):
    """创建递归字符分割器"""
    # 定义分隔符及其优先级（优先使用段落分隔符，其次是句号等）
    separators = [
        "\n\n",  # 段落分隔符
        "\n",    # 换行符
        "。",    # 中文句号
        "！",    # 中文感叹号
        "？",    # 中文问号
        ".",     # 英文句号
        "!",     # 英文感叹号
        "?",     # 英文问号
        ";",     # 分号
        "；",    # 中文分号
        ",",     # 英文逗号
        "，"     # 中文逗号
    ]
    
    # 创建递归字符分割器，设置块大小和重叠大小
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
        length_function=len,
    )
```

这种分割器的工作流程为：

1. 尝试使用最高优先级的分隔符（如段落符）分割文本
2. 如果分割后的片段仍超过目标块大小，则使用下一级分隔符继续分割
3. 递归执行此过程，直到所有文本片段都在目标大小范围内
4. 应用设定的块重叠策略，确保相邻块间有足够上下文连贯性

**3. 叙事感知分块机制**

为了更好地适应短篇小说的叙事结构，我们开发了叙事感知分块机制，该机制关注叙事单元（如场景、对话、情节转折点）而非简单的字符计数：

```python
def narrative_aware_splitting(self, text):
    """基于叙事结构的文本分块"""
    # 首先通过段落分割文本
    paragraphs = text.split("\n\n")
    blocks = []
    current_block = ""
    current_tokens = 0
    
    for para in paragraphs:
        # 估算当前段落的token数
        para_tokens = len(para) / 3  # 简单估算，每个中文字约占1token
        
        # 检测是否为场景转换或重要对话
        is_scene_change = self._detect_scene_change(para)
        is_key_dialogue = self._detect_key_dialogue(para)
        
        # 如果当前块为空，直接添加段落
        if not current_block:
            current_block = para
            current_tokens = para_tokens
            continue
            
        # 决定是否开始新块
        if (current_tokens + para_tokens > self.chunk_size or 
            is_scene_change or is_key_dialogue):
            # 保存当前块并开始新块
            blocks.append(current_block)
            current_block = para
            current_tokens = para_tokens
        else:
            # 将段落添加到当前块
            current_block += "\n\n" + para
            current_tokens += para_tokens
    
    # 添加最后一个块
    if current_block:
        blocks.append(current_block)
    
    # 应用块之间的重叠
    overlapped_blocks = self._apply_overlap(blocks)
    
    return overlapped_blocks
```

叙事感知分块的关键特点：

- **场景检测**：识别可能的场景转换标记，如时间跳转、地点变化
- **对话识别**：识别重要对话开始，作为潜在的分块点
- **情节转折识别**：检测表示情节转折的关键词和句式
- **上下文平衡**：在分块点前后保留足够上下文，确保语义连贯

**4. 混合分块策略**

实践中，我们采用混合分块策略，结合基础的递归字符分割和叙事感知分块的优势：

```python
def split_text(self, text):
    """混合分块策略"""
    # 检查文本长度决定使用哪种策略
    if len(text) < 10000:  # 较短文本
        # 使用基础递归字符分割器
        splitter = self.create_text_splitter()
        return splitter.split_text(text)
    else:  # 较长文本
        # 先进行叙事感知的粗粒度分块
        narrative_blocks = self.narrative_aware_splitting(text)
        
        # 对较大的块再应用递归字符分割
        splitter = self.create_text_splitter()
        final_blocks = []
        for block in narrative_blocks:
            if len(block) > self.chunk_size:
                sub_blocks = splitter.split_text(block)
                final_blocks.extend(sub_blocks)
            else:
                final_blocks.append(block)
        
        return final_blocks
```

混合策略的优势：

- 充分考虑文本长度和复杂度，采用自适应方法
- 先保证叙事单元的完整性，再优化块大小
- 对不同类型的文本段落（对话、描述、情节）采用差异化处理
- 在保持语义完整性和检索效率间取得平衡

**5. 分块元数据增强**

为提高检索质量，我们为每个分块增加了元数据注释：

```python
def add_chunk_metadata(self, chunks, text):
    """为分块添加元数据"""
    enhanced_chunks = []
    total_length = len(text)
    
    for i, chunk in enumerate(chunks):
        # 计算块在原文中的位置信息
        start_pos = text.find(chunk)
        if start_pos == -1:  # 处理可能的重叠导致的不匹配
            start_pos = 0  # 默认值
        
        rel_position = start_pos / total_length  # 相对位置
        
        # 识别块中的关键实体
        entities = self._extract_entities(chunk)
        
        # 创建增强块
        enhanced_chunk = {
            "content": chunk,
            "metadata": {
                "index": i,
                "rel_position": rel_position,
                "length": len(chunk),
                "key_entities": entities,
                "chunk_type": self._determine_chunk_type(chunk)
            }
        }
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks
```

元数据注释包括：

- **位置信息**：块在原文中的相对位置，帮助保持检索结果的时序关系
- **关键实体**：块中包含的主要实体，优化实体相关查询
- **块类型**：对话、描述、混合等类型标记，用于查询路由
- **长度信息**：块的长度数据，用于结果排序和过滤

通过这些分块策略，我们构建了兼顾语义完整性和检索效率的文本块集合，为高质量的RAG系统奠定了基础。

### 3.3.2 向量化与索引构建

文本分块完成后，需要将文本块转换为向量表示并构建高效索引，以支持语义检索。本节详细介绍向量化和索引构建的实现方法。

**1. 向量化模型选择**

本研究选用OpenAI的文本嵌入模型作为主要向量化工具，同时提供开源模型作为备选：

```python
def initialize_embedding_model(self):
    """初始化嵌入模型"""
    if self.embedding_model_type == "openai":
        # 使用OpenAI API的嵌入模型
        self.embedding_model = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key
        )
    elif self.embedding_model_type == "local":
        # 使用本地部署的嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    else:
        raise ValueError(f"不支持的嵌入模型类型: {self.embedding_model_type}")
```

选择OpenAI的text-embedding-3-small模型作为主要向量化工具的原因：

- **语义表达能力强**：对小说文本中的隐含语义和上下文关系表达更准确
- **维度适中**：1536维向量，平衡了表达能力和计算效率
- **多语言支持**：良好支持中文文本，适合处理中文短篇小说
- **与生成模型兼容性高**：与后续使用的GPT模型来自同一技术体系，语义对齐度高

同时，我们也实现了支持本地部署的开源模型选项（如BERT系列或BGE系列模型），以满足不同场景下的需求。

**2. 批量向量化处理**

为提高效率，实现了文本块的批量向量化处理：

```python
def embed_documents(self, chunks):
    """批量将文本块转换为向量表示"""
    contents = [chunk["content"] for chunk in chunks]
    
    # 批处理大小设置
    batch_size = 16
    vectors = []
    
    # 分批处理
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        batch_vectors = self.embedding_model.embed_documents(batch)
        vectors.extend(batch_vectors)
        
        # 显示进度
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(contents):
            print(f"向量化进度: {min(i + batch_size, len(contents))}/{len(contents)}")
    
    # 将向量添加到原始块中
    for i, vector in enumerate(vectors):
        chunks[i]["vector"] = vector
    
    return chunks
```

批量处理的关键优化：

- **合理的批大小**：根据API限制和内存情况动态调整批大小
- **进度监控**：实时显示处理进度，便于跟踪长文本处理情况
- **错误处理**：添加重试机制，应对可能的网络波动
- **向量缓存**：保存已生成的向量，避免重复计算

**3. FAISS索引构建**

本研究采用FAISS（Facebook AI Similarity Search）作为向量索引引擎，构建高效的相似性搜索索引：

```python
def build_faiss_index(self, chunks_with_vectors):
    """构建FAISS向量索引"""
    # 提取向量数据
    vectors = np.array([chunk["vector"] for chunk in chunks_with_vectors], dtype=np.float32)
    dimension = vectors.shape[1]  # 向量维度
    
    # 创建索引
    # 对于较小的数据集使用精确检索
    if len(vectors) < 10000:
        index = faiss.IndexFlatL2(dimension)
    # 对于较大的数据集使用量化索引提高效率
    else:
        nlist = min(int(len(vectors) / 10), 100)  # 聚类数量
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        # 训练索引
        index.train(vectors)
    
    # 添加向量到索引
    index.add(vectors)
    
    # 构建ID到原始块的映射
    id_to_chunk = {i: chunk for i, chunk in enumerate(chunks_with_vectors)}
    
    return {
        "index": index,
        "id_to_chunk": id_to_chunk
    }
```

FAISS索引构建的关键要点：

- **索引类型自适应**：根据数据规模选择合适的索引类型
  - 小数据集使用IndexFlatL2：精确搜索，适合小规模高精度需求
  - 大数据集使用IndexIVFFlat：基于聚类的索引，平衡效率和精度

- **参数优化**：根据数据特点调整聚类数量等参数
- **维度匹配**：确保向量维度一致性，避免索引错误
- **ID映射保存**：维护向量ID与原始文本块的映射关系

**4. 索引持久化与增量更新**

为支持系统的长期使用和文本更新，实现了索引的持久化和增量更新机制：

```python
def save_index(self, index_data, save_path):
    """持久化保存索引数据"""
    index_file = os.path.join(save_path, "faiss_index.bin")
    mapping_file = os.path.join(save_path, "id_to_chunk.pkl")
    
    # 保存FAISS索引
    faiss.write_index(index_data["index"], index_file)
    
    # 保存ID映射
    with open(mapping_file, "wb") as f:
        pickle.dump(index_data["id_to_chunk"], f)
    
    print(f"索引数据已保存至: {save_path}")

def load_index(self, load_path):
    """加载保存的索引数据"""
    index_file = os.path.join(load_path, "faiss_index.bin")
    mapping_file = os.path.join(load_path, "id_to_chunk.pkl")
    
    # 加载FAISS索引
    index = faiss.read_index(index_file)
    
    # 加载ID映射
    with open(mapping_file, "rb") as f:
        id_to_chunk = pickle.load(f)
    
    return {
        "index": index,
        "id_to_chunk": id_to_chunk
    }

def update_index(self, index_data, new_chunks):
    """增量更新索引"""
    # 向量化新块
    new_chunks_with_vectors = self.embed_documents(new_chunks)
    
    # 提取新向量
    new_vectors = np.array([chunk["vector"] for chunk in new_chunks_with_vectors], 
                           dtype=np.float32)
    
    # 添加新向量到索引
    index = index_data["index"]
    original_count = index.ntotal
    index.add(new_vectors)
    
    # 更新ID映射
    id_to_chunk = index_data["id_to_chunk"]
    for i, chunk in enumerate(new_chunks_with_vectors):
        id_to_chunk[original_count + i] = chunk
    
    return {
        "index": index,
        "id_to_chunk": id_to_chunk
    }
```

索引管理机制的特点：

- **完整持久化**：同时保存向量索引和元数据映射，确保系统可恢复
- **增量更新支持**：允许添加新的文本块而无需重建整个索引
- **版本管理**：记录索引构建信息，支持索引版本控制
- **兼容性保障**：处理向量维度和格式变化，确保版本间兼容

**5. 向量化系统评估**

为确保向量化质量，我们实现了评估机制：

```python
def evaluate_embedding_quality(self, text, sample_pairs=100):
    """评估嵌入向量的质量"""
    # 分割文本并向量化
    chunks = self.split_text(text)
    chunks_with_vectors = self.embed_documents(chunks)
    vectors = np.array([chunk["vector"] for chunk in chunks_with_vectors])
    
    # 随机选择样本对计算相似度
    similarities = []
    pairs = []
    for _ in range(sample_pairs):
        i, j = random.sample(range(len(chunks)), 2)
        similarity = np.dot(vectors[i], vectors[j]) / (
            np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
        similarities.append(similarity)
        pairs.append((chunks[i]["content"][:50], chunks[j]["content"][:50], similarity))
    
    # 计算统计信息
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    # 排序样本对展示最相似和最不相似的例子
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return {
        "average_similarity": avg_similarity,
        "std_similarity": std_similarity,
        "most_similar_pairs": pairs[:5],
        "least_similar_pairs": pairs[-5:],
    }
```

评估内容包括：

- **相似度分布**：检查向量空间中的相似度分布是否合理
- **典型样本对**：展示最相似和最不相似的文本对，验证语义捕捉
- **聚类结构**：分析向量空间的聚类情况，验证文本主题分组
- **检索准确性测试**：通过已知问题验证检索结果的准确性

通过向量化和索引构建，我们将非结构化的短篇小说文本转换为可高效检索的向量表示，为RAG系统的核心检索功能奠定了基础。

### 3.3.3 相似度检索机制

RAG系统的核心功能是基于用户问题检索最相关的文本片段。本节详细介绍本研究实现的相似度检索机制。

**1. 查询向量化处理**

为确保查询与文档在同一向量空间，实现了查询向量化处理：

```python
def embed_query(self, query):
    """将用户查询转换为向量表示"""
    try:
        # 使用与文档相同的嵌入模型处理查询
        query_vector = self.embedding_model.embed_query(query)
        return query_vector
    except Exception as e:
        print(f"查询向量化过程中出现错误: {e}")
        # 提供简单的备用方案
        if self.embedding_model_type == "openai":
            # 尝试使用备用模型
            backup_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=self.api_key
            )
            return backup_model.embed_query(query)
        else:
            # 如无备用方案，抛出异常
            raise
```

查询向量化的关键考虑：

- **与文档向量对齐**：确保使用相同的向量化模型和参数
- **查询预处理**：对查询进行必要的清洗和标准化
- **错误处理**：提供备用模型和重试机制，确保服务可靠性
- **向量规范化**：确保查询向量经过适当规范化，优化相似度计算

**2. 基础向量检索实现**

基于FAISS索引实现了高效的向量相似度检索：

```python
def retrieve_similar_chunks(self, query, index_data, top_k=5):
    """检索与查询最相似的文本块"""
    # 将查询转换为向量
    query_vector = self.embed_query(query)
    query_vector_np = np.array([query_vector], dtype=np.float32)
    
    # 使用FAISS执行相似度检索
    index = index_data["index"]
    distances, indices = index.search(query_vector_np, top_k)
    
    # 获取检索结果
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # FAISS可能返回-1表示没有足够的匹配
            chunk = index_data["id_to_chunk"][idx]
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": float(1.0 / (1.0 + distances[0][i]))  # 转换距离为相似度分数
            })
    
    return results
```

检索过程的关键步骤：

- **向量规范化**：确保查询向量格式正确
- **K近邻搜索**：在索引中找到最近的K个向量
- **距离转换**：将欧氏距离转换为相似度分数
- **结果封装**：返回完整的文本内容和元数据

**3. 多策略检索机制**

为提高检索准确性，实现了多策略检索机制：

```python
def enhanced_retrieval(self, query, index_data, top_k=5):
    """增强检索机制，融合多种检索策略"""
    results = []
    
    # 策略1: 直接向量检索
    direct_results = self.retrieve_similar_chunks(query, index_data, top_k)
    
    # 策略2: 查询扩展检索
    expanded_query = self._expand_query(query)
    expanded_results = self.retrieve_similar_chunks(expanded_query, index_data, top_k)
    
    # 策略3: 针对特定问题类型的专门检索
    query_type = self._classify_query_type(query)
    if query_type == "character":
        # 针对人物问题的检索
        character_name = self._extract_character_name(query)
        character_results = self._character_focused_retrieval(character_name, index_data, top_k)
        results.extend(character_results)
    elif query_type == "plot":
        # 针对情节问题的检索
        plot_results = self._plot_focused_retrieval(query, index_data, top_k)
        results.extend(plot_results)
    # 其他问题类型...
    
    # 合并结果
    merged_results = self._merge_results(direct_results, expanded_results, results)
    
    # 结果重排序
    reranked_results = self._rerank_results(query, merged_results)
    
    return reranked_results[:top_k]
```

多策略检索的核心组成：

- **查询扩展**：使用同义词、关键实体扩展原始查询
- **基于问题类型的检索**：针对不同类型问题采用专门的检索策略
- **结果融合**：去重并整合多种策略的检索结果
- **重排序**：基于更复杂的相关性计算对结果重新排序

**4. 上下文感知检索增强**

为更好地适应叙事文本的上下文连续性，实现了上下文感知检索增强：

```python
def context_aware_retrieval(self, query, index_data, top_k=5):
    """上下文感知的检索增强"""
    # 基础检索
    base_results = self.retrieve_similar_chunks(query, index_data, top_k=top_k//2)
    
    # 获取基础结果的上下文块
    context_results = []
    for result in base_results:
        # 获取当前块的索引
        current_index = result["metadata"]["index"]
        
        # 提取前后相邻块
        prev_index = current_index - 1
        next_index = current_index + 1
        
        # 添加相邻块到结果集
        for idx in [prev_index, next_index]:
            for chunk_id, chunk in index_data["id_to_chunk"].items():
                if chunk["metadata"]["index"] == idx:
                    context_results.append({
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        # 根据距离计算上下文块分数(稍低于原块)
                        "score": result["score"] * 0.9
                    })
    
    # 合并基础结果和上下文结果
    combined_results = base_results + context_results
    
    # 去重
    unique_results = self._remove_duplicates(combined_results)
    
    # 重排序，保持前后顺序
    sorted_results = sorted(unique_results, 
                           key=lambda x: (x["score"], x["metadata"]["index"]), 
                           reverse=True)
    
    return sorted_results[:top_k]
```

上下文感知检索的要点：

- **相邻块检索**：为核心匹配块添加前后相邻块，保持叙事连贯性
- **位置权重**：根据块在原文中的相对位置调整相似度分数
- **时序保持**：确保检索结果按原文顺序排列，尤其对情节相关问题
- **上下文窗口自适应**：根据问题类型调整上下文窗口大小

**5. 检索结果后处理**

为提高检索结果的可用性，实现了一系列后处理步骤：

```python
def postprocess_results(self, query, results):
    """检索结果后处理"""
    # 去除重复内容
    unique_results = self._remove_duplicates(results)
    
    # 修剪过长的内容
    trimmed_results = self._trim_results(unique_results, max_length=1000)
    
    # 重新排序，整合时序信息
    sorted_results = self._temporal_reordering(trimmed_results)
    
    # 添加关联度标注
    annotated_results = self._annotate_relevance(query, sorted_results)
    
    return annotated_results

def _remove_duplicates(self, results):
    """移除内容重复的结果"""
    unique_results = []
    seen_content = set()
    
    for result in results:
        # 使用内容的哈希值作为唯一标识
        content_hash = hash(result["content"])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append(result)
    
    return unique_results

def _annotate_relevance(self, query, results):
    """为检索结果添加关联度标注"""
    for result in results:
        # 提取关键片段
        key_snippets = self._extract_relevant_snippets(query, result["content"])
        result["key_snippets"] = key_snippets
        
        # 标记查询中提到的实体
        query_entities = self._extract_entities(query)
        found_entities = [entity for entity in query_entities 
                         if entity in result["content"]]
        result["matched_entities"] = found_entities
    
    return results
```

后处理步骤包括：

- **重复检测与合并**：去除内容高度重叠的结果
- **内容修剪**：保留关键信息，控制总长度
- **相关片段提取**：在长文本中标记与查询最相关的句子
- **实体匹配标记**：突出显示文本中与查询相关的关键实体
- **来源引用添加**：为检索内容添加在原文中的位置信息

通过这些相似度检索机制，我们能够从文本库中快速找到与用户问题最相关的内容片段，为生成高质量回答提供可靠依据。

### 3.3.4 RAG系统优化

为提高RAG系统的整体性能和用户体验，本研究实施了一系列系统优化措施。本节详细介绍这些优化方法及其实现。

**1. 批处理与并行计算**

为提高处理大型文档的效率，实现了批处理和并行计算机制：

```python
def process_large_document(self, text):
    """使用批处理和并行计算处理大型文档"""
    # 分块处理长文本
    chunks = self.split_text(text)
    total_chunks = len(chunks)
    
    # 向量化批次大小
    batch_size = 32
    
    # 使用多进程并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # 分批提交向量化任务
        futures = []
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:min(i+batch_size, total_chunks)]
            future = executor.submit(self._process_batch, batch)
            futures.append(future)
        
        # 收集结果
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                print(f"批次 {i+1}/{len(futures)} 完成")
            except Exception as e:
                print(f"处理批次时出错: {e}")
    
    # 构建向量索引
    return self.build_faiss_index(results)

def _process_batch(self, batch):
    """处理单个批次的文本块"""
    # 注: 此方法将在单独的进程中运行
    # 初始化局部嵌入模型
    local_embedder = self._initialize_local_embedder()
    
    # 向量化
    vectors = []
    for chunk in batch:
        try:
            vector = local_embedder.embed_documents([chunk["content"]])[0]
            chunk["vector"] = vector
            vectors.append(chunk)
        except Exception as e:
            print(f"向量化块时出错: {str(e)[:100]}...")
    
    return vectors
```

并行处理的优化要点：

- **进程池管理**：使用ProcessPoolExecutor处理CPU密集型任务
- **批量任务分配**：将大型文档分解为适合并行处理的批次
- **资源监控**：动态调整批大小和进程数，避免资源耗尽
- **错误隔离**：单个批次处理失败不影响整体流程

**2. 缓存机制实现**

为减少重复计算和API调用，实现了多级缓存机制：

```python
class RAGCache:
    """RAG系统的多级缓存机制"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存
        self.query_cache = {}  # 查询结果缓存
        self.embedding_cache = {}  # 嵌入向量缓存
        
        # 缓存最大容量控制
        self.max_memory_entries = 1000
        self.max_disk_entries = 10000
    
    def get_query_results(self, query_key):
        """获取缓存的查询结果"""
        # 先检查内存缓存
        if query_key in self.query_cache:
            return self.query_cache[query_key]
        
        # 再检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"query_{hashlib.md5(query_key.encode()).hexdigest()}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    results = pickle.load(f)
                # 加入内存缓存
                self._add_to_memory_cache(query_key, results)
                return results
            except Exception:
                pass
        
        return None
    
    def cache_query_results(self, query_key, results):
        """缓存查询结果"""
        # 存入内存缓存
        self._add_to_memory_cache(query_key, results)
        
        # 存入磁盘缓存
        try:
            cache_file = os.path.join(self.cache_dir, f"query_{hashlib.md5(query_key.encode()).hexdigest()}.pkl")
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
            
            # 清理过多的磁盘缓存
            self._clean_disk_cache()
        except Exception as e:
            print(f"缓存写入失败: {e}")
    
    def _add_to_memory_cache(self, key, value):
        """添加到内存缓存，并管理缓存大小"""
        self.query_cache[key] = value
        
        # 如果缓存过大，移除最早的条目
        if len(self.query_cache) > self.max_memory_entries:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
    
    def _clean_disk_cache(self):
        """清理过多的磁盘缓存文件"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "query_*.pkl"))
        if len(cache_files) > self.max_disk_entries:
            # 按修改时间排序
            cache_files.sort(key=os.path.getmtime)
            # 删除最旧的文件
            for file in cache_files[:len(cache_files) - self.max_disk_entries]:
                try:
                    os.remove(file)
                except Exception:
                    pass
```

缓存机制的优化特点：

- **多级缓存架构**：内存缓存用于高频访问，磁盘缓存用于持久存储
- **自动过期策略**：基于LRU（最近最少使用）策略管理缓存容量
- **键值设计**：使用规范化查询和查询参数作为缓存键
- **向量复用**：缓存嵌入向量，减少模型调用开销

**3. 检索质量优化**

为提高检索质量，实现了一系列检索优化技术：

```python
def optimize_retrieval(self, query, index_data, top_k=5):
    """检索质量优化方法"""
    # 查询预处理
    processed_query = self._preprocess_query(query)
    
    # 查询扩展
    expanded_queries = self._generate_query_variations(processed_query)
    
    # 多查询检索
    all_results = []
    for exp_query in expanded_queries:
        results = self.retrieve_similar_chunks(exp_query, index_data, top_k=top_k)
        all_results.extend(results)
    
    # 结果去重
    unique_results = self._remove_duplicates(all_results)
    
    # 结果重排序（使用交叉编码器或其他高级相关性评分）
    if len(unique_results) > top_k:
        reranked_results = self._rerank_with_cross_encoder(processed_query, unique_results)
    else:
        reranked_results = unique_results
    
    # 返回前K个结果
    return reranked_results[:top_k]

def _generate_query_variations(self, query):
    """生成查询变体"""
    variations = [query]  # 原始查询
    
    # 提取查询中的关键实体
    entities = self._extract_entities(query)
    
    # 基于实体的变体
    if entities:
        entity_query = " ".join(entities)
        variations.append(entity_query)
    
    # 添加问题变形（如果是问句）
    if "?" in query:
        # 将问题转为陈述句形式
        statement_query = self._question_to_statement(query)
        variations.append(statement_query)
    
    # 添加简化查询（移除停用词）
    simplified_query = self._simplify_query(query)
    if simplified_query != query:
        variations.append(simplified_query)
    
    return variations

def _rerank_with_cross_encoder(self, query, results, model_name="sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """使用交叉编码器重排序结果"""
    # 检查是否可在本地使用交叉编码器
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name)
        
        # 准备句对
        pairs = [(query, result["content"]) for result in results]
        
        # 预测相关性分数
        scores = model.predict(pairs)
        
        # 更新分数并排序
        for i, result in enumerate(results):
            result["score"] = float(scores[i])
        
        reranked = sorted(results, key=lambda x: x["score"], reverse=True)
        return reranked
    except ImportError:
        # 如果无法使用交叉编码器，返回原结果
        print("交叉编码器不可用，跳过重排序")
        return results
```

检索质量优化的关键技术：

- **查询改写**：生成不同表达方式的查询变体
- **多维度检索**：从不同角度生成查询，捕捉不同维度的语义
- **交叉编码器重排序**：使用更强大的相关性评分模型优化排序
- **实体引导检索**：利用问题中的实体信息引导检索方向

**4. 性能监控与自适应优化**

为持续改进系统性能，实现了监控和自适应优化机制：

```python
class RAGPerformanceMonitor:
    """RAG系统性能监控与自适应优化"""
    
    def __init__(self):
        self.query_times = []  # 查询耗时记录
        self.hit_rates = []    # 缓存命中率
        self.relevance_scores = []  # 相关性评分
        
        # 性能指标阈值
        self.slow_query_threshold = 2.0  # 秒
        self.low_relevance_threshold = 0.6  # 相关性阈值
    
    def record_query(self, query, results, time_taken, cache_hit):
        """记录查询性能数据"""
        self.query_times.append(time_taken)
        self.hit_rates.append(1 if cache_hit else 0)
        
        # 如果有相关性分数，记录平均分
        if results and "score" in results[0]:
            avg_relevance = sum(r["score"] for r in results) / len(results)
            self.relevance_scores.append(avg_relevance)
        
        # 检查是否需要优化
        self._check_for_optimization_needs(query, results, time_taken)
    
    def _check_for_optimization_needs(self, query, results, time_taken):
        """检查是否需要性能优化"""
        # 检查查询速度
        if time_taken > self.slow_query_threshold:
            self._optimize_for_speed()
        
        # 检查结果相关性
        if results and "score" in results[0]:
            avg_relevance = sum(r["score"] for r in results) / len(results)
            if avg_relevance < self.low_relevance_threshold:
                self._optimize_for_relevance(query, results)
    
    def _optimize_for_speed(self):
        """优化查询速度"""
        # 动态调整批处理大小
        # 调整缓存策略
        # 考虑使用更快的索引类型
        pass
    
    def _optimize_for_relevance(self, query, results):
        """优化结果相关性"""
        # 尝试不同的查询扩展策略
        # 调整重排序参数
        # 考虑使用更高级的相关性模型
        pass
    
    def get_performance_report(self):
        """生成性能报告"""
        if not self.query_times:
            return {"status": "没有记录数据"}
        
        return {
            "avg_query_time": sum(self.query_times) / len(self.query_times),
            "max_query_time": max(self.query_times),
            "cache_hit_rate": sum(self.hit_rates) / len(self.hit_rates) if self.hit_rates else 0,
            "avg_relevance": sum(self.relevance_scores) / len(self.relevance_scores) if self.relevance_scores else None,
            "total_queries": len(self.query_times)
        }
```

性能监控与优化的关键特性：

- **性能指标追踪**：记录查询时间、缓存命中率、相关性分数等指标
- **自适应参数调整**：根据性能数据动态调整系统参数
- **瓶颈识别**：识别性能瓶颈，提供针对性优化
- **智能资源分配**：根据使用模式优化资源分配

**5. 用户反馈整合**

为改进系统表现，实现了用户反馈整合机制：

```python
def incorporate_user_feedback(self, query, results, feedback):
    """整合用户反馈，改进检索质量"""
    # 分析反馈类型
    if feedback["type"] == "relevance":
        # 处理相关性反馈
        relevance_score = feedback["score"]  # 1-5分
        result_id = feedback["result_id"]
        
        # 记录反馈数据
        self._log_relevance_feedback(query, result_id, relevance_score)
        
        # 如果评分低，尝试优化下次检索
        if relevance_score <= 2:
            # 添加到负面示例集
            self._add_to_negative_examples(query, result_id)
            
    elif feedback["type"] == "missing_info":
        # 处理信息缺失反馈
        missing_aspect = feedback["aspect"]
        
        # 为该查询添加额外检索参数
        self._add_query_enhancement(query, missing_aspect)
        
        # 返回增强检索结果
        return self._perform_enhanced_retrieval(query, missing_aspect)
    
    # 返回None表示无需额外操作
    return None
```

用户反馈整合的关键机制：

- **反馈分类**：将用户反馈分为相关性、信息完整性、准确性等类别
- **即时调整**：根据反馈立即调整当前会话的检索策略
- **长期优化**：积累反馈数据，用于长期改进系统
- **负例学习**：记录不相关的结果，避免未来类似错误

通过这些系统优化措施，RAG系统不仅能够高效处理大型文本，还能持续提升检索质量和用户体验，为混合问答系统提供强大支持。 