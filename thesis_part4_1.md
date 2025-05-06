# 第四章 实验设计与结果分析

本章详细介绍针对KG-RAG混合短篇小说问答系统的实验设计与结果分析。首先阐述实验环境与数据集，其次介绍实验设计方案，然后分析系统性能的实验结果，最后讨论系统优缺点及应用价值。

## 4.1 实验环境与数据集

### 4.1.1 实验环境配置

**1. 硬件环境**

本研究中使用的实验硬件环境配置如下：

| 配置项 | 规格 |
|-------|------|
| 处理器 | Intel Core i9-12900K (16核24线程) |
| 内存 | 64GB DDR5 4800MHz |
| 存储 | 2TB NVMe SSD |
| 图形处理器 | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| 网络 | 千兆以太网 |

对于密集型计算任务，特别是知识图谱构建和向量检索，系统还可以扩展到云端服务器：

```python
def initialize_cloud_resources():
    """初始化云资源用于大规模实验"""
    # 配置云服务连接
    cloud_config = {
        "provider": "aws",
        "instance_type": "p3.2xlarge",  # 包含Tesla V100 GPU
        "region": "us-west-2",
        "storage": "500GB"
    }
    
    # 创建实验环境
    client = CloudClient(cloud_config)
    instance = client.create_instance()
    
    # 部署实验代码和依赖
    instance.install_dependencies([
        "torch==2.0.1",
        "transformers==4.30.2",
        "networkx==3.1",
        "faiss-gpu==1.7.2",
        "spacy==3.6.0",
        "langchain==0.0.235"
    ])
    
    return instance
```

**2. 软件环境**

软件环境配置如下表所示：

| 软件类别 | 名称及版本 |
|---------|-----------|
| 操作系统 | Ubuntu 22.04 LTS |
| 编程语言 | Python 3.10.12 |
| 深度学习框架 | PyTorch 2.0.1, Transformers 4.30.2 |
| 数据库 | Neo4j 5.9.0 (图数据库), MongoDB 6.0 (文档存储) |
| 向量数据库 | FAISS 1.7.2, Milvus 2.2.8 |
| 自然语言处理 | SpaCy 3.6.0, NLTK 3.8.1 |
| 大模型接口 | OpenAI API, Anthropic API |
| 其他库 | LangChain 0.0.235, NetworkX 3.1 |

**3. 模型配置**

本研究采用了以下模型配置：

```python
def load_model_configurations():
    """加载实验中使用的模型配置"""
    model_configs = {
        "embedding_models": {
            "primary": {
                "name": "text-embedding-3-small",
                "provider": "OpenAI",
                "dimensions": 1536,
                "batch_size": 16
            },
            "alternative": {
                "name": "bge-large-zh-v1.5",
                "provider": "BAAI",
                "dimensions": 1024,
                "batch_size": 32
            }
        },
        "llm_models": {
            "primary": {
                "name": "gpt-4-turbo",
                "provider": "OpenAI",
                "context_window": 128000,
                "temperature": 0.7
            },
            "alternative": {
                "name": "claude-3-opus-20240229",
                "provider": "Anthropic",
                "context_window": 200000,
                "temperature": 0.7
            }
        },
        "kg_models": {
            "entity_extraction": {
                "name": "zh_core_web_trf",
                "provider": "SpaCy",
                "version": "3.6.0"
            },
            "relation_extraction": {
                "name": "bert-base-chinese",
                "provider": "Hugging Face",
                "fine_tuned": True
            }
        }
    }
    
    return model_configs
```

### 4.1.2 实验数据集构建

**1. 数据集来源**

本研究使用了多源数据集，包括：

- **中国当代短篇小说集**：从多个文学期刊和出版物中精选的100篇短篇小说
- **跨文化短篇小说译作**：包含30篇世界知名短篇小说的中文译本
- **网络原创短篇小说**：从正规网络文学平台收集的50篇优质原创短篇作品

数据集的基本统计信息如表4-1所示：

| 数据集类别 | 数量 | 平均字数 | 总字数 | 发表年代跨度 |
|-----------|-----|---------|-------|------------|
| 当代短篇小说 | 100篇 | 8,500字 | 850,000字 | 1990-2023年 |
| 跨文化译作 | 30篇 | 12,000字 | 360,000字 | 1900-2020年 |
| 网络原创作品 | 50篇 | 6,000字 | 300,000字 | 2010-2023年 |
| 总计 | 180篇 | 8,400字 | 1,510,000字 | 1900-2023年 |

**表4-1 实验数据集基本统计信息**

**2. 数据集预处理**

对收集的原始文本数据进行了以下预处理：

```python
def preprocess_dataset(raw_text_collection):
    """对原始文本数据进行预处理"""
    processed_dataset = []
    
    for text_item in raw_text_collection:
        # 文本清洗
        cleaned_text = text_cleaning(text_item["content"])
        
        # 结构化处理
        structured_text = {
            "id": text_item["id"],
            "title": text_item["title"],
            "author": text_item["author"],
            "publication_year": text_item["year"],
            "content": cleaned_text,
            "category": text_item["category"],
            "word_count": len(cleaned_text),
            "processed_date": datetime.now().isoformat()
        }
        
        # 基础分析
        analysis_result = perform_basic_analysis(cleaned_text)
        structured_text.update({
            "paragraphs_count": analysis_result["paragraphs"],
            "sentence_count": analysis_result["sentences"],
            "entity_count": analysis_result["entities"],
            "dialogue_ratio": analysis_result["dialogue_ratio"]
        })
        
        processed_dataset.append(structured_text)
    
    return processed_dataset

def text_cleaning(text):
    """文本清洗函数"""
    # 移除不必要的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 规范化标点符号
    text = normalize_punctuation(text)
    
    # 修复明显的OCR错误
    text = fix_common_ocr_errors(text)
    
    # 统一全角/半角字符
    text = unify_full_half_width(text)
    
    # 段落规范化
    text = normalize_paragraphs(text)
    
    return text
```

预处理过程包括：
- 文本清洗与规范化（标点符号、空白字符处理）
- 段落和句子分割
- 基本实体识别与标记
- 对话与叙述部分识别
- 文本结构分析与标记

**3. 问答对构建**

为评估系统性能，我们构建了多种类型的问答对：

```python
def generate_qa_pairs(processed_dataset):
    """基于处理后的数据集生成问答对"""
    qa_pairs = []
    
    for text_item in processed_dataset:
        # 基于规则的问题生成
        rule_based_qa = generate_rule_based_questions(text_item)
        
        # 基于模板的问题生成
        template_qa = generate_template_questions(text_item)
        
        # 基于LLM的问题生成
        llm_generated_qa = generate_llm_questions(text_item)
        
        # 人工编写的黄金问答对
        if text_item["id"] in gold_standard_texts:
            manual_qa = load_manual_qa_pairs(text_item["id"])
            qa_pairs.extend(manual_qa)
        
        # 添加自动生成的问答对
        qa_pairs.extend(rule_based_qa)
        qa_pairs.extend(template_qa)
        qa_pairs.extend(llm_generated_qa)
    
    # 对问题进行去重和质量过滤
    filtered_qa_pairs = filter_qa_pairs(qa_pairs)
    
    # 问题分类与标注
    categorized_qa_pairs = categorize_qa_pairs(filtered_qa_pairs)
    
    return categorized_qa_pairs
```

问答对按照以下类型进行分类：

| 问题类型 | 说明 | 示例 | 数量 |
|---------|-----|------|-----|
| 事实型问题 | 询问文本中明确陈述的事实 | "小说中主角的职业是什么？" | 650个 |
| 关系型问题 | 询问实体间的关系 | "李明和张红是什么关系？" | 480个 |
| 情节型问题 | 询问故事情节发展 | "主角为什么决定离开家乡？" | 420个 |
| 推理型问题 | 需要推理才能回答 | "根据文中描述，主角对生活的态度是什么？" | 380个 |
| 综合型问题 | 需要整合多处信息 | "小说中有哪些象征手法及其含义？" | 310个 |
| 对比型问题 | 要求对比分析 | "小说开头和结尾的场景有什么呼应关系？" | 260个 |

**表4-2 问答对类型分布**

最终构建的问答对总数为2,500对，其中包含600对人工编写的高质量问答对作为评估基准。

### 4.1.3 评估数据集划分

将构建的问答对划分为训练集、验证集和测试集：

```python
def split_dataset(qa_pairs, split_ratio=[0.7, 0.1, 0.2]):
    """划分数据集为训练集、验证集和测试集"""
    # 打乱数据
    random.shuffle(qa_pairs)
    
    # 计算分割点
    train_end = int(len(qa_pairs) * split_ratio[0])
    val_end = train_end + int(len(qa_pairs) * split_ratio[1])
    
    # 划分数据集
    train_set = qa_pairs[:train_end]
    val_set = qa_pairs[train_end:val_end]
    test_set = qa_pairs[val_end:]
    
    # 确保每种问题类型在测试集中都有代表
    test_set = ensure_question_type_coverage(test_set, qa_pairs)
    
    # 确保人工编写的问答对主要分布在测试集中
    manual_qa_pairs = [qa for qa in qa_pairs if qa["source"] == "manual"]
    test_set = ensure_manual_qa_in_test(test_set, manual_qa_pairs)
    
    return {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }
```

最终的数据集划分如下：

| 数据集 | 问答对数量 | 人工编写问答对比例 | 小说覆盖数量 |
|-------|----------|-----------------|-----------|
| 训练集 | 1,750对 | 10% | 150篇 |
| 验证集 | 250对 | 30% | 90篇 |
| 测试集 | 500对 | 80% | 130篇 |

**表4-3 数据集划分情况**

特别注意，测试集中包含了不同难度级别的问题，以全面评估系统性能：

| 难度级别 | 比例 | 特点 |
|---------|-----|------|
| 简单 | 25% | 直接可从文本中找到答案 |
| 中等 | 50% | 需要整合2-3处信息 |
| 困难 | 25% | 需要复杂推理或整合多处信息 |

**表4-4 测试集问题难度分布** 