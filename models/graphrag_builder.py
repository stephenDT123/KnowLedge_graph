"""
完整的GraphRAG构建模块
集成LLM API和Embedding模型，支持实体识别、关系抽取和向量索引
"""

import asyncio
import json
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel

from config import GraphRAGConfig, get_default_config, load_config_from_env
from pdf_data_processor import process_pdf

try:
    from openai import AsyncOpenAI, OpenAI
    from openai.types import CreateEmbeddingResponse
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


# ==================== 数据模型 ====================

@dataclass
class Document:
    """文档数据模型"""
    id: str
    title: str
    source_type: str
    source_path: str
    text: str
    metadata: Dict[str, str]


@dataclass
class TextUnit:
    """文本单元数据模型"""
    id: str
    document_id: str
    index: int
    text: str
    tokens: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class Entity:
    """实体数据模型"""
    id: str
    name: str
    entity_type: str
    frequency: int
    description: str
    document_ids: List[str]
    text_unit_ids: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class Relationship:
    """关系数据模型"""
    id: str
    source: str
    target: str
    relation_type: str
    weight: int
    evidence_text_unit_ids: List[str]


@dataclass
class Community:
    """社区数据模型"""
    id: str
    entity_ids: List[str]
    relationship_ids: List[str]
    title: str
    summary: str


@dataclass
class GraphRAGDataset:
    """GraphRAG完整数据集"""
    documents: List[Document]
    text_units: List[TextUnit]
    entities: List[Entity]
    relationships: List[Relationship]
    communities: List[Community]
    config: GraphRAGConfig


# ==================== Pydantic模型用于LLM响应 ====================

class EntityExtractionResult(BaseModel):
    """实体抽取结果"""
    entities: List[dict]  # 简化为List[dict]，避免复杂的类型注解
    confidence: float


class RelationExtractionResult(BaseModel):
    """关系抽取结果"""
    relationships: List[dict]  # 简化为List[dict]，避免复杂的类型注解
    confidence: float


class CommunitySummaryResult(BaseModel):
    """社区摘要结果"""
    summary: str
    highlights: List[str]
    teaching_suggestions: List[str]
    learning_paths: List[str]


# ==================== LLM客户端 ====================

class LLMClient:
    """LLM客户端抽象类"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.llm_config = config.llm_config
        self._client = None
        
    async def extract_entities(self, text: str) -> EntityExtractionResult:
        """使用LLM抽取实体"""
        prompt = self.config.entity_extraction_prompt.format(text=text)
        response = await self._call_llm(prompt)
        return EntityExtractionResult.model_validate_json(response)
    
    async def extract_relations(self, text: str, entities: List[str]) -> RelationExtractionResult:
        """使用LLM抽取关系"""
        prompt = self.config.relation_extraction_prompt.format(
            text=text, 
            entities=", ".join(entities)
        )
        response = await self._call_llm(prompt)
        return RelationExtractionResult.model_validate_json(response)
    
    async def generate_community_summary(self, title: str, entities: List[str], 
                                        relationships: List[str]) -> CommunitySummaryResult:
        """生成社区摘要"""
        prompt = self.config.community_summary_prompt.format(
            title=title,
            entities=", ".join(entities),
            relationships=", ".join(relationships)
        )
        response = await self._call_llm(prompt)
        return CommunitySummaryResult.model_validate_json(response)
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM API（子类实现）"""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI客户端"""
    
    def __init__(self, config: GraphRAGConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI包未安装，请运行: pip install openai")
        
        self._client = AsyncOpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """调用OpenAI API"""
        try:
            response = await self._client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": "你是一个教育知识图谱专家，擅长从文本中识别教育领域的实体和关系。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            # 返回空结果
            return json.dumps({"entities": [], "confidence": 0.0})


class DeepSeekClient(LLMClient):
    """DeepSeek客户端"""
    
    def __init__(self, config: GraphRAGConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI包未安装，请运行: pip install openai")
        
        # DeepSeek使用OpenAI兼容的API
        self._client = AsyncOpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url or "https://api.deepseek.com"
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """调用DeepSeek API"""
        try:
            # 确保提示词包含"json"以便使用JSON响应格式
            enhanced_prompt = f"请以JSON格式返回结果。{prompt}"
            
            response = await self._client.chat.completions.create(
                model=self.llm_config.model or "deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个教育知识图谱专家，擅长从文本中识别教育领域的实体和关系。"},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            # 返回空结果
            return json.dumps({"entities": [], "confidence": 0.0})


# ==================== Embedding客户端 ====================

class EmbeddingClient:
    """Embedding客户端抽象类"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.embedding_config = config.embedding_config
        
    async def embed_text(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        raise NotImplementedError
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI Embedding客户端"""
    
    def __init__(self, config: GraphRAGConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI包未安装，请运行: pip install openai")
        
        self._client = OpenAI(
            api_key=self.embedding_config.api_key,
            base_url=self.embedding_config.base_url
        )
    
    async def embed_text(self, text: str) -> List[float]:
        """生成单个文本嵌入"""
        try:
            response: CreateEmbeddingResponse = self._client.embeddings.create(
                model=self.embedding_config.model,
                input=text,
                dimensions=self.embedding_config.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI Embedding调用失败: {e}")
            # 返回零向量
            dims = self.embedding_config.dimensions or 1536
            return [0.0] * dims
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        embeddings = []
        # 分批处理，避免API限制
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response: CreateEmbeddingResponse = self._client.embeddings.create(
                    model=self.embedding_config.model,
                    input=batch,
                    dimensions=self.embedding_config.dimensions
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"批量Embedding调用失败: {e}")
                # 为失败的批次生成零向量
                dims = self.embedding_config.dimensions or 1536
                embeddings.extend([[0.0] * dims] * len(batch))
        
        return embeddings


class AliyunEmbeddingClient(EmbeddingClient):
    """阿里云Embedding客户端"""
    
    def __init__(self, config: GraphRAGConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI包未安装，请运行: pip install openai")
        
        # 阿里云使用OpenAI兼容的API
        self._client = OpenAI(
            api_key=self.embedding_config.api_key,
            base_url=self.embedding_config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    async def embed_text(self, text: str) -> List[float]:
        """生成单个文本嵌入"""
        try:
            response = self._client.embeddings.create(
                model=self.embedding_config.model or "text-embedding-v2",
                input=text,
                dimensions=self.embedding_config.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"阿里云Embedding调用失败: {e}")
            # 返回零向量
            dims = self.embedding_config.dimensions or 1536
            return [0.0] * dims
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        embeddings = []
        # 分批处理，避免API限制
        batch_size = 50  # 阿里云可能有不同的限制
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self.embedding_config.model or "text-embedding-v2",
                    input=batch,
                    dimensions=self.embedding_config.dimensions
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"批量阿里云Embedding调用失败: {e}")
                # 为失败的批次生成零向量
                dims = self.embedding_config.dimensions or 1536
                embeddings.extend([[0.0] * dims] * len(batch))
        
        return embeddings


# ==================== 核心构建函数 ====================

def _new_id(prefix: str) -> str:
    """生成新的ID"""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """将文本切分为重叠的块"""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        # 尝试在句子边界处切分
        if end < len(cleaned):
            # 查找最近的句子结束符
            sentence_end = max(
                cleaned.rfind(".", start, end),
                cleaned.rfind("。", start, end),
                cleaned.rfind("!", start, end),
                cleaned.rfind("！", start, end),
                cleaned.rfind("?", start, end),
                cleaned.rfind("？", start, end)
            )
            if sentence_end > start + max_chars // 2:  # 确保块不会太小
                end = sentence_end + 1
        
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    
    return chunks


def tokenize_zh(text: str) -> List[str]:
    """中文分词"""
    if JIEBA_AVAILABLE:
        return [token.strip() for token in jieba.cut(text) if token and token.strip()]
    else:
        # 简单的基于正则的分词
        return [m.group(0) for m in re.finditer(r"[\u4e00-\u9fa5A-Za-z0-9_]{2,}", text)]


async def build_documents_from_sources(
    pdf_paths: Optional[Iterable[str]] = None,
    video_paths: Optional[Iterable[str]] = None,
    config: Optional[GraphRAGConfig] = None
) -> List[Document]:
    """从源文件构建文档"""
    if config is None:
        config = get_default_config()
    
    documents: List[Document] = []
    
    # 处理PDF文件
    for pdf_path in pdf_paths or []:
        try:
            result = process_pdf(
                pdf_path, 
                enable_ocr=config.enable_ocr,
                tesseract_cmd=config.tesseract_cmd
            )
            
            documents.append(
                Document(
                    id=_new_id("doc"),
                    title=Path(pdf_path).stem,
                    source_type="pdf",
                    source_path=str(pdf_path),
                    text=result["raw_text"],
                    metadata={
                        "tokens_count": len(result["tokens"]),
                        "filtered_tokens_count": len(result["filtered_tokens"]),
                        "has_ocr": config.enable_ocr
                    }
                )
            )
        except Exception as e:
            print(f"处理PDF文件失败 {pdf_path}: {e}")
    
    # 处理视频文件（简化版）
    for video_path in video_paths or []:
        try:
            # 这里应该调用视频处理模块
            # 暂时使用占位文本
            documents.append(
                Document(
                    id=_new_id("doc"),
                    title=Path(video_path).stem,
                    source_type="video",
                    source_path=str(video_path),
                    text=f"视频文件: {Path(video_path).name} (待处理)",
                    metadata={"status": "pending"}
                )
            )
        except Exception as e:
            print(f"处理视频文件失败 {video_path}: {e}")
    
    return documents


async def build_text_units(
    documents: List[Document], 
    config: GraphRAGConfig,
    embedding_client: Optional[EmbeddingClient] = None
) -> List[TextUnit]:
    """构建文本单元并生成嵌入向量"""
    text_units: List[TextUnit] = []
    
    for document in documents:
        chunks = chunk_text(
            document.text, 
            max_chars=config.chunk_size,
            overlap=config.chunk_overlap
        )
        
        for idx, chunk in enumerate(chunks):
            text_units.append(
                TextUnit(
                    id=_new_id("tu"),
                    document_id=document.id,
                    index=idx,
                    text=chunk,
                    tokens=tokenize_zh(chunk)
                )
            )
    
    # 生成嵌入向量
    if embedding_client:
        texts = [unit.text for unit in text_units]
        embeddings = await embedding_client.embed_batch(texts)
        
        for unit, embedding in zip(text_units, embeddings):
            unit.embedding = embedding
    
    return text_units


async def build_entities_and_relationships(
    text_units: List[TextUnit],
    config: GraphRAGConfig,
    llm_client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None
) -> Tuple[List[Entity], List[Relationship]]:
    """构建实体和关系"""
    entity_stats: Dict[str, Dict] = defaultdict(lambda: {
        "id": _new_id("ent"),
        "name": "",
        "entity_type": "",
        "frequency": 0,
        "description": "",
        "document_ids": set(),
        "text_unit_ids": set()
    })
    
    relationships: Dict[Tuple[str, str, str], Dict] = {}
    
    # 处理每个文本单元
    for unit in text_units:
        # 方法1：使用LLM抽取实体和关系
        if llm_client:
            try:
                # 抽取实体
                entity_result = await llm_client.extract_entities(unit.text)
                
                # 更新实体统计
                if hasattr(entity_result, 'entities') and entity_result.entities:
                    for item in entity_result.entities:
                        # 确保item是字典并且包含必要的键
                        if isinstance(item, dict) and "name" in item and "type" in item:
                            name = item["name"]
                            entity_type = item["type"]
                            
                            if name not in entity_stats:
                                entity_stats[name]["name"] = name
                                entity_stats[name]["entity_type"] = entity_type
                            
                            entity_stats[name]["frequency"] += 1
                            entity_stats[name]["document_ids"].add(unit.document_id)
                            entity_stats[name]["text_unit_ids"].add(unit.id)
                        else:
                            print(f"⚠️  实体数据格式不正确: {item}")
                
                # 抽取关系
                if hasattr(entity_result, 'entities') and entity_result.entities:
                    entity_names = [item["name"] for item in entity_result.entities 
                                   if isinstance(item, dict) and "name" in item]
                    if entity_names:
                        relation_result = await llm_client.extract_relations(unit.text, entity_names)
                        
                        if hasattr(relation_result, 'relationships') and relation_result.relationships:
                            for rel in relation_result.relationships:
                                if isinstance(rel, dict) and "source" in rel and "target" in rel and "type" in rel:
                                    key = (rel["source"], rel["target"], rel["type"])
                                    if key not in relationships:
                                        relationships[key] = {
                                            "id": _new_id("rel"),
                                            "source": rel["source"],
                                            "target": rel["target"],
                                            "relation_type": rel["type"],
                                            "weight": 0,
                                            "evidence_text_unit_ids": set()
                                        }
                                    relationships[key]["weight"] += 1
                                    relationships[key]["evidence_text_unit_ids"].add(unit.id)
                                else:
                                    print(f"⚠️  关系数据格式不正确: {rel}")
                        
            except Exception as e:
                print(f"LLM实体关系抽取失败: {e}")
        
        # 方法2：基于规则的实体识别（备用）
        else:
            tokens = tokenize_zh(unit.text)
            for token in tokens:
                if len(token) >= 2:
                    if token not in entity_stats:
                        entity_stats[token]["name"] = token
                        entity_stats[token]["entity_type"] = "Concept"  # 默认类型
                    
                    entity_stats[token]["frequency"] += 1
                    entity_stats[token]["document_ids"].add(unit.document_id)
                    entity_stats[token]["text_unit_ids"].add(unit.id)
    
    # 构建实体对象
    entities: List[Entity] = []
    for name, stats in entity_stats.items():
        if stats["frequency"] > 0:  # 只保留出现过的实体
            entities.append(
                Entity(
                    id=stats["id"],
                    name=name,
                    entity_type=stats["entity_type"],
                    frequency=stats["frequency"],
                    description=f"{name} 是《Python程序设计》课程中的 {stats['entity_type']} 节点。",
                    document_ids=sorted(stats["document_ids"]),
                    text_unit_ids=sorted(stats["text_unit_ids"])
                )
            )
    
    # 生成实体嵌入向量
    if embedding_client:
        entity_texts = [entity.name for entity in entities]
        entity_embeddings = await embedding_client.embed_batch(entity_texts)
        
        for entity, embedding in zip(entities, entity_embeddings):
            entity.embedding = embedding
    
    # 构建关系对象
    entity_id_by_name = {entity.name: entity.id for entity in entities}
    relationship_objects: List[Relationship] = []
    
    for key, rel_data in relationships.items():
        source_name, target_name, rel_type = key
        if source_name in entity_id_by_name and target_name in entity_id_by_name:
            relationship_objects.append(
                Relationship(
                    id=rel_data["id"],
                    source=entity_id_by_name[source_name],
                    target=entity_id_by_name[target_name],
                    relation_type=rel_type,
                    weight=rel_data["weight"],
                    evidence_text_unit_ids=sorted(rel_data["evidence_text_unit_ids"])
                )
            )
    
    # 排序
    entities.sort(key=lambda e: (-e.frequency, e.name))
    relationship_objects.sort(key=lambda r: (-r.weight, r.relation_type, r.source, r.target))
    
    return entities, relationship_objects


async def build_communities(
    entities: List[Entity],
    relationships: List[Relationship],
    config: GraphRAGConfig,
    llm_client: Optional[LLMClient] = None
) -> List[Community]:
    """构建知识社区"""
    # 构建邻接表
    adjacency: Dict[str, set] = defaultdict(set)
    rel_ids_by_entity: Dict[str, List[str]] = defaultdict(list)
    entity_by_id = {entity.id: entity for entity in entities}
    
    for rel in relationships:
        adjacency[rel.source].add(rel.target)
        adjacency[rel.target].add(rel.source)
        rel_ids_by_entity[rel.source].append(rel.id)
        rel_ids_by_entity[rel.target].append(rel.id)
    
    # 查找连通分量（社区）
    visited = set()
    communities: List[Community] = []
    
    for entity in entities:
        if entity.id in visited:
            continue
        
        # BFS查找连通分量
        queue = [entity.id]
        component_entity_ids = []
        component_rel_ids = set()
        
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component_entity_ids.append(current)
            
            # 收集相关关系
            for rel_id in rel_ids_by_entity[current]:
                component_rel_ids.add(rel_id)
            
            # 添加邻居
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        # 构建社区
        component_entities = [entity_by_id[eid] for eid in component_entity_ids]
        component_entities.sort(key=lambda e: (-e.frequency, e.name))
        
        # 生成社区标题
        title = "、".join(e.name for e in component_entities[:3]) or "课程知识社区"
        
        # 生成社区摘要
        summary = ""
        if llm_client:
            try:
                entity_names = [e.name for e in component_entities]
                rel_descriptions = [f"{rel.relation_type}({rel.source}→{rel.target})" 
                                  for rel in relationships if rel.id in component_rel_ids]
                
                summary_result = await llm_client.generate_community_summary(
                    title, entity_names, rel_descriptions
                )
                summary = summary_result.summary
            except Exception as e:
                print(f"生成社区摘要失败: {e}")
                summary = f"该社区包含 {len(component_entities)} 个实体和 {len(component_rel_ids)} 条关系。"
        else:
            summary = f"该社区包含 {len(component_entities)} 个实体和 {len(component_rel_ids)} 条关系。"
        
        communities.append(
            Community(
                id=_new_id("community"),
                entity_ids=sorted(component_entity_ids),
                relationship_ids=sorted(component_rel_ids),
                title=title,
                summary=summary
            )
        )
    
    return communities


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    """写入JSONL文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def build_graphrag_dataset(
    output_dir: Union[str, os.PathLike],
    pdf_paths: Optional[Iterable[str]] = None,
    video_paths: Optional[Iterable[str]] = None,
    config: Optional[GraphRAGConfig] = None
) -> GraphRAGDataset:
    """构建完整的GraphRAG数据集"""
    if config is None:
        config = get_default_config()
    
    # 初始化客户端
    llm_client = None
    embedding_client = None
    
    try:
        # 根据配置创建LLM客户端
        provider = config.llm_config.provider.value
        if provider == "openai":
            llm_client = OpenAIClient(config)
        elif provider == "deepseek":
            llm_client = DeepSeekClient(config)
        else:
            print(f"不支持的LLM提供商: {provider}，将使用规则方法")
        
        # 根据配置创建Embedding客户端
        embedding_provider = config.embedding_config.provider.value
        if embedding_provider == "openai":
            embedding_client = OpenAIEmbeddingClient(config)
        elif embedding_provider == "aliyun":
            embedding_client = AliyunEmbeddingClient(config)
        else:
            print(f"不支持的Embedding提供商: {embedding_provider}，将不使用嵌入向量")
    except Exception as e:
        print(f"客户端初始化失败，将使用规则方法: {e}")
    
    # 构建文档
    print("正在构建文档...")
    documents = await build_documents_from_sources(
        pdf_paths=pdf_paths,
        video_paths=video_paths,
        config=config
    )
    
    # 构建文本单元
    print("正在构建文本单元和嵌入向量...")
    text_units = await build_text_units(documents, config, embedding_client)
    
    # 构建实体和关系
    print("正在抽取实体和关系...")
    entities, relationships = await build_entities_and_relationships(
        text_units, config, llm_client, embedding_client
    )
    
    # 构建社区
    print("正在构建知识社区...")
    communities = await build_communities(entities, relationships, config, llm_client)
    
    # 保存数据
    print("正在保存数据...")
    output_path = Path(output_dir)
    
    _write_jsonl(output_path / "documents.jsonl", [asdict(doc) for doc in documents])
    _write_jsonl(output_path / "text_units.jsonl", [asdict(unit) for unit in text_units])
    _write_jsonl(output_path / "entities.jsonl", [asdict(entity) for entity in entities])
    _write_jsonl(output_path / "relationships.jsonl", [asdict(rel) for rel in relationships])
    _write_jsonl(output_path / "communities.jsonl", [asdict(comm) for comm in communities])
    
    # 保存配置
    config_dict = {
        "llm_config": {
            "provider": config.llm_config.provider.value,
            "model": config.llm_config.model,
            "temperature": config.llm_config.temperature
        },
        "embedding_config": {
            "provider": config.embedding_config.provider.value,
            "model": config.embedding_config.model,
            "dimensions": config.embedding_config.dimensions
        },
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap
    }
    
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    return GraphRAGDataset(
        documents=documents,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        communities=communities,
        config=config
    )


async def build_from_project_defaults() -> GraphRAGDataset:
    """使用项目默认配置构建GraphRAG数据集"""
    config = load_config_from_env()
    
    base_dir = Path(__file__).resolve().parent
    default_pdf = base_dir / "附件：学员操作手册.pdf"
    pdf_paths = [str(default_pdf)] if default_pdf.exists() else []
    
    return await build_graphrag_dataset(
        output_dir=base_dir / config.output_dir,
        pdf_paths=pdf_paths,
        video_paths=[],
        config=config
    )


# ==================== 命令行接口 ====================

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建GraphRAG知识图谱数据集")
    parser.add_argument("--pdf", nargs="+", help="PDF文件路径列表")
    parser.add_argument("--video", nargs="+", help="视频文件路径列表")
    parser.add_argument("--output", default="graphrag_output", help="输出目录")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--enable-ocr", action="store_true", help="启用OCR")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        # 从文件加载配置
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        # 这里需要将config_data转换为GraphRAGConfig
        # 简化处理：使用默认配置
        config = get_default_config()
    else:
        config = load_config_from_env()
    
    if args.enable_ocr:
        config.enable_ocr = True
    
    # 构建数据集
    dataset = await build_graphrag_dataset(
        output_dir=args.output,
        pdf_paths=args.pdf,
        video_paths=args.video,
        config=config
    )
    
    # 输出摘要
    summary = {
        "documents": len(dataset.documents),
        "text_units": len(dataset.text_units),
        "entities": len(dataset.entities),
        "relationships": len(dataset.relationships),
        "communities": len(dataset.communities),
        "output_dir": str(Path(args.output).resolve())
    }
    
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return dataset


if __name__ == "__main__":
    asyncio.run(main())