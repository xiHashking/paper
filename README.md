# 增强型问答系统（带知识图谱和RAG）

这是一个集成了知识图谱和检索增强生成（RAG）技术的增强型问答系统，使用OpenAI API进行嵌入和问答。

## 功能特点

1. **知识图谱构建**：使用spaCy和关键词提取方法从文本中自动提取实体和关系，构建可视化知识图谱
2. **RAG系统**：使用OpenAI的嵌入模型和FAISS向量索引实现检索增强生成
3. **OpenAI集成**：使用OpenAI的大语言模型进行最终回答生成
4. **多种问答方式**：支持直接使用OpenAI、基于知识图谱的问答、基于RAG的问答和RAG与知识图谱结合的混合问答

## 安装与设置

### 环境要求

- Python 3.8+
- 相关依赖库（见requirements.txt）

### 安装步骤

1. 克隆或下载本项目代码
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装spaCy中文模型：

```bash
python -m spacy download zh_core_web_sm
```

4. 设置API密钥：
   - 编辑.env文件并填入OpenAI API密钥：
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 使用方法

### 准备数据

将你的文本文件放入`data`目录下，项目已包含示例文本文件。

### 运行程序

1. 执行主程序：
```bash
python main.py
```

程序将执行以下操作：
1. 读取文本数据
2. 使用spaCy构建知识图谱（保存在output目录）
3. 使用关键词方法构建知识图谱（保存在output目录）
4. 构建RAG系统
5. 启动交互式问答界面

### 问答交互

在交互式界面中：
- 输入任意问题
- 系统将分别使用四种方法回答：
  1. 直接使用OpenAI
  2. 基于知识图谱
  3. 基于RAG
  4. RAG与知识图谱结合（提供最全面的答案）
- 输入"exit"或"quit"退出

## 项目结构

- `openai_client.py`: OpenAI API客户端
- `knowledge_graph.py`: 知识图谱构建与查询（使用spaCy和关键词方法）
- `rag_system.py`: 检索增强生成系统（使用OpenAI嵌入和FAISS）
- `main.py`: 主程序
- `data/`: 存放文本数据
- `output/`: 存放输出文件（如知识图谱可视化）

## 故障排除

如果遇到安装依赖问题：
1. 确保使用Python 3.8+版本
2. 更新pip: `python -m pip install --upgrade pip`
3. 如果spaCy中文模型安装失败，可以尝试：`pip install spacy && python -m spacy download zh_core_web_sm`
4. 如果遇到unstructured安装问题，可以尝试：`pip install "unstructured[all-docs]"`

## 扩展与自定义

### 添加更多文档

将更多文本文件放入`data`目录，系统将自动处理。支持的文件类型：
- .txt（默认）
- 可在代码中扩展支持其他类型

### 自定义知识图谱提取方法

知识图谱构建支持两种方法：
- "spacy": 使用spaCy自然语言处理库提取三元组
- "keywords": 使用关键词提取方法提取三元组

### 调整RAG参数

可以调整以下参数：
- 文本分块大小（chunk_size）
- 分块重叠量（chunk_overlap）
- 检索文档数量（top_k）

### 混合问答方法

系统提供了一种混合问答方法，结合了RAG和知识图谱的优势：
- 同时使用检索到的文档内容和知识图谱中的结构化关系
- 提供更全面、更准确的回答
- 适合需要综合理解的复杂问题