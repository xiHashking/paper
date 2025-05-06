## 4.2 实验设计与基准系统

### 4.2.1 实验方案设计

**1. 实验目标与研究问题**

本研究的实验设计围绕以下核心研究问题展开：

1. 知识图谱增强和检索增强生成各自对短篇小说问答的效果如何？
2. 本研究提出的KG-RAG混合方法是否优于单一技术方法？
3. 不同类型问题下，各方法的优势和局限性是什么？
4. 混合系统的信息融合策略对问答质量的影响如何？
5. 系统在资源消耗与回答质量之间的权衡如何？

基于这些研究问题，设计了一系列系统对比实验和消融实验。

**2. 实验设计框架**

实验框架的总体设计如图4-1所示：

```
+-----------------------------------+
|            实验数据集              |
+-----------------------------------+
                |
                v
+-----------------------------------+
|        系统构建与对比实验           |
+-----------------------------------+
      /        |         |        \
     v         v         v         v
+---------+ +---------+ +---------+ +---------+
| 传统LLM | |   KG    | |   RAG   | | KG-RAG  |
|  系统   | |  系统   | |  系统   | |  系统   |
+---------+ +---------+ +---------+ +---------+
      \        |         |        /
       v       v         v       v
+-----------------------------------+
|      性能指标评估与比较            |
+-----------------------------------+
                |
                v
+-----------------------------------+
|          消融实验                  |
+-----------------------------------+
      /        |         |        \
     v         v         v         v
+---------+ +---------+ +---------+ +---------+
| 融合策略 | | 提示工程 | | 分块策略 | | 其他参数 |
|  变体   | |  变体   | |  变体   | |  变体   |
+---------+ +---------+ +---------+ +---------+
      \        |         |        /
       v       v         v       v
+-----------------------------------+
|      案例分析与误差分析            |
+-----------------------------------+
```

**图4-1 实验框架设计**

**3. 评估指标体系**

为全面评估系统性能，建立了多维度的评估指标体系：

```python
def evaluate_system_performance(system, test_dataset):
    """评估系统性能的多维度指标"""
    results = {}
    
    # 回答质量评估
    answer_quality = evaluate_answer_quality(system, test_dataset)
    results["answer_quality"] = answer_quality
    
    # 信息检索效果评估
    retrieval_metrics = evaluate_retrieval_performance(system, test_dataset)
    results["retrieval_metrics"] = retrieval_metrics
    
    # 系统效率评估
    efficiency_metrics = evaluate_system_efficiency(system, test_dataset)
    results["efficiency_metrics"] = efficiency_metrics
    
    # 按问题类型分组的性能
    performance_by_question_type = evaluate_by_question_type(system, test_dataset)
    results["by_question_type"] = performance_by_question_type
    
    # 系统鲁棒性评估
    robustness_metrics = evaluate_system_robustness(system, test_dataset)
    results["robustness_metrics"] = robustness_metrics
    
    return results
```

具体评估指标如表4-5所示：

| 评估维度 | 具体指标 | 计算方法 |
|---------|---------|---------|
| **回答质量** | 内容完整性 | 评估回答覆盖问题所需方面的比例 |
| | 实体覆盖率 | 评估回答中包含问题相关的关键实体的程度 |
| | 事实准确率 | 评估回答中的事实陈述准确性 |
| | 知识库一致性 | 评估回答与知识库的整体一致性 |
| **信息检索** | Precision@k | 前k个检索结果中相关结果比例 |
| | Recall@k | 相关结果中被检索到的比例 |
| | MRR | 平均倒数排名 |
| | MAP | 平均准确率均值 |
| **系统效率** | 平均响应时间 | 从提问到回答的平均时间 |
| | 内存使用 | 系统运行期间的内存占用 |
| | API调用次数 | 处理每个问题的API调用数量 |
| | 索引建立时间 | 构建知识图谱和向量索引的时间 |
| **鲁棒性** | 噪声抗干扰性 | 在有噪声数据下的性能下降程度 |
| | 跨文档泛化性 | 在未见文档上的性能表现 |
| | 复杂问题处理能力 | 处理多跳推理问题的成功率 |

**表4-5 评估指标体系**

**4. 人工评估设计**

除自动评估外，还设计了严格的人工评估流程：

```python
def human_evaluation_process(system_outputs, evaluators):
    """人工评估流程"""
    # 生成评估任务
    evaluation_tasks = []
    for question_id, system_answers in system_outputs.items():
        # 打乱不同系统的回答顺序
        shuffled_answers = shuffle_system_answers(system_answers)
        
        task = {
            "question_id": question_id,
            "question": get_question_by_id(question_id),
            "context": get_context_by_question_id(question_id),
            "answers": shuffled_answers,
            "evaluation_form": create_evaluation_form()
        }
        evaluation_tasks.append(task)
    
    # 分配评估任务给评估者
    assigned_tasks = assign_tasks_to_evaluators(evaluation_tasks, evaluators)
    
    # 收集评估结果
    evaluation_results = collect_evaluation_results(assigned_tasks)
    
    # 分析评估结果
    analysis = analyze_human_evaluation(evaluation_results)
    
    return analysis
```

人工评估采用了以下设计：

- **三重盲评**：评估者不知道哪个回答来自哪个系统
- **多维度评分**：对每个回答在5个维度上进行1-5分的评分
- **定性反馈**：收集评估者对每个回答的文字评价
- **偏好排序**：要求评估者对不同系统的回答进行排序
- **评估者构成**：包括NLP专家、文学领域专家和普通用户

### 4.2.2 基准系统实现

为进行对比实验，实现了三个基准系统：传统LLM系统、仅知识图谱系统和仅RAG系统。

**1. 传统LLM基准系统**

传统LLM系统直接使用大语言模型回答问题，不进行额外知识增强：

```python
class TraditionalLLMSystem:
    """传统LLM基准系统"""
    
    def __init__(self, config):
        self.config = config
        self.llm_provider = LLMProvider(config["llm_config"])
        self.context_size = config.get("context_size", 2000)
    
    def initialize(self):
        """系统初始化"""
        # 仅初始化LLM接口
        self.system_info = {
            "name": "Traditional LLM System",
            "llm_model": self.config["llm_config"]["model_name"],
            "initialized_at": datetime.now().isoformat()
        }
        return True
    
    def process_document(self, document):
        """处理文档（仅保存文档内容）"""
        # 如果文档太长，只保留前部分
        if len(document["content"]) > self.context_size:
            self.document = {
                "title": document["title"],
                "content": document["content"][:self.context_size],
                "truncated": True
            }
        else:
            self.document = {
                "title": document["title"],
                "content": document["content"],
                "truncated": False
            }
        
        return {"status": "success", "truncated": self.document["truncated"]}
    
    def answer_question(self, question):
        """回答问题"""
        start_time = time.time()
        
        # 构建提示
        prompt = self._build_prompt(question)
        
        # 调用LLM生成回答
        response = self.llm_provider.generate(prompt)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": response,
            "system": self.system_info["name"],
            "processing_time": processing_time,
            "document_title": self.document["title"]
        }
    
    def _build_prompt(self, question):
        """构建提示"""
        return f"""你是一个智能问答助手。请根据下面的短篇小说内容回答问题。

标题：{self.document["title"]}

内容：
{self.document["content"]}

问题：{question}

请直接回答问题，不需要重复问题。如果小说内容中没有相关信息，请说明无法回答。"""
```

传统LLM系统的主要限制：

- 受上下文窗口大小限制，无法处理完整的长篇小说
- 无外部知识增强，仅依赖模型参数中的知识
- 无法验证答案的准确性
- 容易产生幻觉

**2. 仅知识图谱系统**

仅知识图谱系统使用图谱查询结果辅助LLM回答问题：

```python
class KnowledgeGraphOnlySystem:
    """仅知识图谱系统"""
    
    def __init__(self, config):
        self.config = config
        self.llm_provider = LLMProvider(config["llm_config"])
        self.kg_module = KnowledgeGraphModule(config["kg_config"])
    
    def initialize(self):
        """系统初始化"""
        kg_init_result = self.kg_module.initialize()
        
        self.system_info = {
            "name": "Knowledge Graph Only System",
            "llm_model": self.config["llm_config"]["model_name"],
            "kg_info": kg_init_result,
            "initialized_at": datetime.now().isoformat()
        }
        
        return kg_init_result["status"] == "success"
    
    def process_document(self, document):
        """处理文档，构建知识图谱"""
        self.document = document
        kg_result = self.kg_module.build_knowledge_graph(document["content"])
        
        return {
            "status": kg_result["status"],
            "graph_info": {
                "nodes": kg_result["node_count"],
                "edges": kg_result["edge_count"],
                "build_time": kg_result["build_time"]
            }
        }
    
    def answer_question(self, question):
        """基于知识图谱回答问题"""
        start_time = time.time()
        
        # 分析问题类型
        question_analysis = self.kg_module.analyze_question(question)
        
        # 查询知识图谱
        kg_query_result = self.kg_module.query_knowledge_graph(
            question, 
            question_analysis
        )
        
        # 构建提示
        prompt = self._build_prompt(question, kg_query_result)
        
        # 调用LLM生成回答
        response = self.llm_provider.generate(prompt)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": response,
            "kg_query_result": kg_query_result,
            "system": self.system_info["name"],
            "processing_time": processing_time,
            "document_title": self.document["title"]
        }
    
    def _build_prompt(self, question, kg_query_result):
        """构建基于知识图谱结果的提示"""
        kg_info_text = self._format_kg_results(kg_query_result)
        
        return f"""你是一个基于知识图谱的智能问答助手。请根据下面的知识图谱查询结果回答问题。

问题：{question}

知识图谱提供的信息：
{kg_info_text}

请根据知识图谱提供的信息回答问题。如果信息不足，请明确指出。回答应该简洁、准确，直接基于提供的知识图谱信息。"""
```

仅知识图谱系统的特点：

- 专注于实体和关系提取，结构化信息处理能力强
- 对关系型和事实型问题有明显优势
- 对叙述性内容和上下文依赖性问题支持有限
- 构建质量严重依赖于图谱构建效果

**3. 仅RAG系统**

仅RAG系统利用向量检索增强LLM回答能力：

```python
class RAGOnlySystem:
    """仅RAG系统"""
    
    def __init__(self, config):
        self.config = config
        self.llm_provider = LLMProvider(config["llm_config"])
        self.rag_module = RAGModule(config["rag_config"])
    
    def initialize(self):
        """系统初始化"""
        rag_init_result = self.rag_module.initialize()
        
        self.system_info = {
            "name": "RAG Only System",
            "llm_model": self.config["llm_config"]["model_name"],
            "embedding_model": self.config["rag_config"]["embedding_model"],
            "initialized_at": datetime.now().isoformat()
        }
        
        return rag_init_result["status"] == "success"
    
    def process_document(self, document):
        """处理文档，创建向量索引"""
        self.document = document
        rag_result = self.rag_module.process_document(document["content"])
        
        return {
            "status": rag_result["status"],
            "indexing_info": {
                "chunks": rag_result["chunk_count"],
                "indexing_time": rag_result["indexing_time"],
                "chunk_size": self.config["rag_config"]["chunk_size"]
            }
        }
    
    def answer_question(self, question):
        """基于RAG回答问题"""
        start_time = time.time()
        
        # 检索相关文本片段
        retrieval_results = self.rag_module.retrieve(
            question, 
            top_k=self.config["rag_config"].get("top_k", 5)
        )
        
        # 构建提示
        prompt = self._build_prompt(question, retrieval_results)
        
        # 调用LLM生成回答
        response = self.llm_provider.generate(prompt)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": response,
            "retrieval_results": retrieval_results,
            "system": self.system_info["name"],
            "processing_time": processing_time,
            "document_title": self.document["title"]
        }
    
    def _build_prompt(self, question, retrieval_results):
        """构建基于检索结果的提示"""
        context_text = self._format_retrieval_results(retrieval_results)
        
        return f"""你是一个基于检索增强的智能问答助手。请根据下面检索到的相关文本片段回答问题。

问题：{question}

相关文本片段：
{context_text}

请根据提供的文本片段回答问题。如果信息不足，请明确指出。回答应该简洁、准确，直接基于提供的文本信息。"""
```

仅RAG系统的特点：

- 能够处理任意长度的文档
- 对文本的语义理解和上下文把握能力强
- 对描述性和叙述性问题有优势
- 检索质量严重依赖于分块和相似度计算效果

### 4.2.3 实验实施流程

**1. 系统训练与优化流程**

各系统的训练与优化流程如下：

```python
def train_and_optimize_systems(training_data, configs):
    """训练与优化各系统"""
    systems = {}
    
    # 初始化各系统
    systems["traditional"] = TraditionalLLMSystem(configs["traditional"])
    systems["kg_only"] = KnowledgeGraphOnlySystem(configs["kg_only"])
    systems["rag_only"] = RAGOnlySystem(configs["rag_only"])
    systems["hybrid"] = KGRAGHybridSystem(configs["hybrid"])
    
    # 系统训练与优化循环
    for system_name, system in systems.items():
        print(f"Training and optimizing {system_name} system...")
        
        # 初始化系统
        system.initialize()
        
        # 在验证集上进行参数优化
        best_params = optimize_system_parameters(
            system=system,
            validation_data=training_data["validation"],
            parameter_grid=parameter_grids[system_name],
            optimization_metric="f1_score"
        )
        
        # 更新系统配置
        system.update_config(best_params)
        
        # 在训练集上进行完整训练/微调
        if hasattr(system, "train"):
            system.train(training_data["train"])
        
        # 保存优化后的系统
        save_system(system, f"models/{system_name}_optimized")
        
        # 记录优化结果
        optimization_results[system_name] = {
            "best_params": best_params,
            "validation_performance": evaluate_on_dataset(
                system, training_data["validation"]
            )
        }
    
    return systems, optimization_results
```

参数优化采用了网格搜索结合贝叶斯优化的方法，针对不同系统优化不同的参数集：

| 系统类型 | 主要优化参数 |
|---------|------------|
| 传统LLM | 温度、最大生成长度、提示模板 |
| 仅知识图谱 | 实体关系阈值、查询深度、图谱复杂度 |
| 仅RAG | 块大小、块重叠、检索数量、重排序阈值 |
| KG-RAG混合 | 融合权重、查询策略、混合方法、提示优化 |

**2. 评估执行流程**

实验评估的完整执行流程如下：

```python
def execute_evaluation(systems, test_data):
    """执行系统评估"""
    results = {}
    
    # 对每个系统进行评估
    for system_name, system in systems.items():
        print(f"Evaluating {system_name} system...")
        system_results = []
        
        # 对每个测试文档进行处理
        for doc_id, document in test_data["documents"].items():
            # 处理文档
            system.process_document(document)
            
            # 回答该文档对应的问题
            doc_questions = get_questions_for_document(test_data["questions"], doc_id)
            for question in doc_questions:
                # 生成回答
                answer_result = system.answer_question(question["text"])
                
                # 评估回答
                evaluation = evaluate_answer(
                    answer_result["answer"],
                    question["reference_answers"],
                    document["content"]
                )
                
                # 记录结果
                result_entry = {
                    "question_id": question["id"],
                    "question": question["text"],
                    "answer": answer_result["answer"],
                    "reference_answers": question["reference_answers"],
                    "metrics": evaluation,
                    "processing_time": answer_result.get("processing_time", 0),
                    "document_id": doc_id
                }
                system_results.append(result_entry)
        
        # 计算整体指标
        aggregated_metrics = aggregate_metrics(system_results)
        
        # 记录系统结果
        results[system_name] = {
            "detailed_results": system_results,
            "aggregated_metrics": aggregated_metrics
        }
    
    # 比较各系统性能
    systems_comparison = compare_systems(results)
    
    return {
        "detailed_results": results,
        "systems_comparison": systems_comparison
    }
```

**3. 消融实验设计**

为深入分析系统各组件的贡献，设计了一系列消融实验：

```python
def ablation_experiments(base_system, test_data):
    """消融实验"""
    ablation_results = {}
    
    # 1. 融合策略消融
    fusion_variants = create_fusion_strategy_variants(base_system)
    ablation_results["fusion_strategy"] = evaluate_variants(
        fusion_variants, test_data
    )
    
    # 2. 提示工程消融
    prompt_variants = create_prompt_engineering_variants(base_system)
    ablation_results["prompt_engineering"] = evaluate_variants(
        prompt_variants, test_data
    )
    
    # 3. 文本分块策略消融
    chunking_variants = create_chunking_strategy_variants(base_system)
    ablation_results["chunking_strategy"] = evaluate_variants(
        chunking_variants, test_data
    )
    
    # 4. 知识图谱组件消融
    kg_variants = create_kg_component_variants(base_system)
    ablation_results["kg_components"] = evaluate_variants(
        kg_variants, test_data
    )
    
    # 5. 混合问答模块消融
    qa_module_variants = create_qa_module_variants(base_system)
    ablation_results["qa_module"] = evaluate_variants(
        qa_module_variants, test_data
    )
    
    return ablation_results
```

消融实验包括以下变体：

1. **融合策略变体**：
   - 简单拼接融合
   - 加权融合
   - 自适应动态融合
   - 阶段性融合

2. **提示工程变体**：
   - 基础提示
   - 结构化提示
   - 思维链提示
   - 无提示优化

3. **文本分块变体**：
   - 固定大小分块
   - 语义感知分块
   - 叙事感知分块
   - 混合分块策略

4. **知识图谱组件变体**：
   - 移除实体属性
   - 简化关系类型
   - 降低图谱复杂度
   - 移除图谱推理

5. **混合问答模块变体**：
   - 串行处理
   - 并行处理
   - 移除冲突解决
   - 移除质量评估 