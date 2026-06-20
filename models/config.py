"""
GraphRAG项目配置模块
支持多种LLM API和Embedding模型配置
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMProvider(Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


class EmbeddingProvider(Enum):
    """支持的Embedding模型提供商"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    ALIYUN = "aliyun"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    def __post_init__(self):
        # 从环境变量获取API密钥
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.AZURE_OPENAI:
                self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            elif self.provider == LLMProvider.GOOGLE:
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider == LLMProvider.DEEPSEEK:
                self.api_key = os.getenv("DEEPSEEK_API_KEY")


@dataclass
class EmbeddingConfig:
    """Embedding模型配置"""
    provider: EmbeddingProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    
    def __post_init__(self):
        # 从环境变量获取API密钥
        if self.api_key is None:
            if self.provider == EmbeddingProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == EmbeddingProvider.AZURE_OPENAI:
                self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            elif self.provider == EmbeddingProvider.COHERE:
                self.api_key = os.getenv("COHERE_API_KEY")
            elif self.provider == EmbeddingProvider.VOYAGE:
                self.api_key = os.getenv("VOYAGE_API_KEY")


@dataclass
class GraphRAGConfig:
    """GraphRAG完整配置"""
    llm_config: LLMConfig
    embedding_config: EmbeddingConfig
    chunk_size: int = 500
    chunk_overlap: int = 50
    enable_ocr: bool = False
    tesseract_cmd: Optional[str] = None
    output_dir: str = "graphrag_output"
    
    # 实体抽取配置
    entity_extraction_prompt: str = """
    请从以下文本中识别教育领域的实体，特别是《Python程序设计》课程相关的概念、技能、评测和资源。
    
    文本内容：
    {text}
    
    请识别以下类型的实体：
    1. 概念（Concept）：编程概念、语法元素、数据结构等
    2. 技能（Skill）：编程技能、调试能力、算法设计等  
    3. 评测（Assessment）：实验、作业、测试、考试等
    4. 资源（Resource）：教材、视频、示例、案例等
    
    请以JSON格式返回结果，包含以下字段：
    - entities: 实体列表，每个实体包含 name（名称）和 type（类型）
    - confidence: 整体置信度（0-1）
    """
    
    # 关系抽取配置
    relation_extraction_prompt: str = """
    请从以下文本中识别教育领域实体之间的关系，特别是《Python程序设计》课程中的先修关系、包含关系等。
    
    文本内容：
    {text}
    
    已知实体：
    {entities}
    
    请识别以下类型的关系：
    1. PREREQUISITE（先修关系）：A是B的基础
    2. CONTAINS（包含关系）：A包括B
    3. DEPENDS_ON（依赖关系）：A依赖B
    4. USED_FOR（用途关系）：A用于B
    5. IMPLEMENTS（实现关系）：A实现B
    
    请以JSON格式返回结果，包含以下字段：
    - relationships: 关系列表，每个关系包含 source（源实体）、target（目标实体）、type（关系类型）
    - confidence: 整体置信度（0-1）
    """
    
    # 社区摘要配置
    community_summary_prompt: str = """
    请为以下知识社区生成教育领域的摘要报告。
    
    社区标题：{title}
    社区实体：{entities}
    社区关系：{relationships}
    
    请生成包含以下内容的摘要：
    1. 社区主题概述
    2. 核心知识点分析
    3. 教学应用建议
    4. 学习路径推荐
    
    请以JSON格式返回结果，包含以下字段：
    - summary: 社区摘要
    - highlights: 关键亮点列表
    - teaching_suggestions: 教学建议
    - learning_paths: 学习路径推荐
    """


def get_default_config() -> GraphRAGConfig:
    """获取默认配置"""
    # 使用DeepSeek LLM模型
    llm_config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model="deepseek-chat",  # DeepSeek模型
        temperature=0.1,
        max_tokens=2000,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"  # DeepSeek API地址
    )
    
    # 使用阿里云Embedding模型
    embedding_config = EmbeddingConfig(
        provider=EmbeddingProvider.ALIYUN,
        model="text-embedding-v2",  # 阿里云文本嵌入模型
        dimensions=1536,
        api_key=os.getenv("ALIYUN_EMBEDDING_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云兼容模式API
    )
    
    return GraphRAGConfig(
        llm_config=llm_config,
        embedding_config=embedding_config,
        chunk_size=500,
        chunk_overlap=50,
        enable_ocr=False,
        output_dir="graphrag_output"
    )


def load_config_from_env() -> GraphRAGConfig:
    """从环境变量加载配置"""
    # 读取LLM提供商
    llm_provider_str = os.getenv("GRAPH_RAG_LLM_PROVIDER", "openai").lower()
    llm_provider = LLMProvider(llm_provider_str)
    
    # 读取Embedding提供商
    embedding_provider_str = os.getenv("GRAPH_RAG_EMBEDDING_PROVIDER", "openai").lower()
    embedding_provider = EmbeddingProvider(embedding_provider_str)
    
    llm_config = LLMConfig(
        provider=llm_provider,
        model=os.getenv("GRAPH_RAG_LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("GRAPH_RAG_LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("GRAPH_RAG_LLM_MAX_TOKENS", "2000")),
        base_url=os.getenv("GRAPH_RAG_LLM_BASE_URL")
    )
    
    embedding_config = EmbeddingConfig(
        provider=embedding_provider,
        model=os.getenv("GRAPH_RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=int(os.getenv("GRAPH_RAG_EMBEDDING_DIMENSIONS", "1536")) if os.getenv("GRAPH_RAG_EMBEDDING_DIMENSIONS") else None,
        base_url=os.getenv("GRAPH_RAG_EMBEDDING_BASE_URL")
    )
    
    return GraphRAGConfig(
        llm_config=llm_config,
        embedding_config=embedding_config,
        chunk_size=int(os.getenv("GRAPH_RAG_CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("GRAPH_RAG_CHUNK_OVERLAP", "50")),
        enable_ocr=os.getenv("GRAPH_RAG_ENABLE_OCR", "false").lower() == "true",
        tesseract_cmd=os.getenv("TESSERACT_CMD"),
        output_dir=os.getenv("GRAPH_RAG_OUTPUT_DIR", "graphrag_output")
    )
