from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

try:
    from neo4j import GraphDatabase
except ImportError as exc:
    raise ImportError("缺少 neo4j 依赖，请先执行: pip install neo4j") from exc


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""
    if not file_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def chunked(rows: List[Dict[str, Any]], size: int = 200) -> Iterable[List[Dict[str, Any]]]:
    """将列表切分为批次，避免单次事务过大。"""
    for index in range(0, len(rows), size):
        yield rows[index:index + size]


def sanitize_rel_type(rel_type: str) -> str:
    """将关系类型转换为安全的 Neo4j 关系名。"""
    raw = (rel_type or "RELATED_TO").upper().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^A-Z0-9_]", "_", raw).strip("_")
    if not cleaned:
        return "RELATED_TO"
    if cleaned[0].isdigit():
        return f"REL_{cleaned}"
    return cleaned


def sanitize_label(label: str) -> str:
    """将实体类型转换为安全的 Neo4j 标签。"""
    raw = (label or "Concept").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not cleaned:
        return "Concept"
    if cleaned[0].isdigit():
        return f"Label_{cleaned}"
    return cleaned


def _is_primitive(value: Any) -> bool:
    """判断值是否为 Neo4j 可直接接受的基本类型。"""
    return isinstance(value, (str, int, float, bool))


def to_neo4j_property_value(value: Any) -> Any:
    """将复杂对象转换为 Neo4j 可接受的属性值。"""
    if value is None:
        return None

    if _is_primitive(value):
        return value

    if isinstance(value, (list, tuple)):
        if all(item is None or _is_primitive(item) for item in value):
            return [item for item in value if item is not None]
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def compact_properties(row: Dict[str, Any], include_embeddings: bool) -> Dict[str, Any]:
    """过滤 None，并按需移除 embedding。"""
    result: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if key == "embedding" and not include_embeddings:
            continue
        converted = to_neo4j_property_value(value)
        if converted is None:
            continue
        result[key] = converted
    return result


def normalize_neo4j_uri(uri: str) -> str:
    """将本地单机版 Neo4j 的 neo4j:// 地址自动转换为 bolt://。"""
    parsed = urlparse(uri)
    if parsed.scheme != "neo4j":
        return uri

    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1"}:
        return uri.replace("neo4j://", "bolt://", 1)

    return uri


class Neo4jGraphImporter:
    """将 GraphRAG 输出导入 Neo4j。"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        include_embeddings: bool = False,
    ) -> None:
        self.uri = normalize_neo4j_uri(uri)
        self.user = user
        self.password = password
        self.database = database
        self.include_embeddings = include_embeddings
        self.driver = GraphDatabase.driver(self.uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def import_from_directory(self, input_dir: Path, clear_existing: bool = False) -> Dict[str, int]:
        """从 GraphRAG 输出目录导入数据。"""
        documents = read_jsonl(input_dir / "documents.jsonl")
        text_units = read_jsonl(input_dir / "text_units.jsonl")
        entities = read_jsonl(input_dir / "entities.jsonl")
        relationships = read_jsonl(input_dir / "relationships.jsonl")
        communities = read_jsonl(input_dir / "communities.jsonl")

        with self.driver.session(database=self.database) as session:
            if clear_existing:
                self._write(session, self._clear_graph)

            for batch in chunked(documents):
                self._write(session, self._merge_documents, batch, self.include_embeddings)

            for batch in chunked(text_units):
                self._write(session, self._merge_text_units, batch, self.include_embeddings)

            for batch in chunked(entities):
                self._write(session, self._merge_entities, batch, self.include_embeddings)

            for batch in chunked(communities):
                self._write(session, self._merge_communities, batch)

            self._write(session, self._link_document_text_units, text_units)
            self._write(session, self._link_document_entities, entities)
            self._write(session, self._link_text_unit_entities, entities)
            self._write(session, self._link_community_entities, communities)

            grouped_relationships: Dict[str, List[Dict[str, Any]]] = {}
            for row in relationships:
                rel_type = sanitize_rel_type(str(row.get("relation_type", "RELATED_TO")))
                grouped_relationships.setdefault(rel_type, []).append(row)

            for rel_type, rows in grouped_relationships.items():
                for batch in chunked(rows):
                    self._write(
                        session,
                        self._merge_entity_relationships,
                        rel_type,
                        batch,
                    )

        return {
            "documents": len(documents),
            "text_units": len(text_units),
            "entities": len(entities),
            "relationships": len(relationships),
            "communities": len(communities),
        }

    @staticmethod
    def _write(session: Any, func: Any, *args: Any) -> Any:
        """兼容 neo4j v4/v5 的写事务 API。"""
        if hasattr(session, "execute_write"):
            return session.execute_write(func, *args)
        return session.write_transaction(func, *args)

    @staticmethod
    def _clear_graph(tx: Any) -> None:
        tx.run(
            """
            MATCH (n)
            WHERE n:Document OR n:TextUnit OR n:Entity OR n:Community
            DETACH DELETE n
            """
        )

    @staticmethod
    def _merge_documents(tx: Any, rows: List[Dict[str, Any]], include_embeddings: bool) -> None:
        payload = [compact_properties(row, include_embeddings) for row in rows]
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d += row
            """,
            rows=payload,
        )

    @staticmethod
    def _merge_text_units(tx: Any, rows: List[Dict[str, Any]], include_embeddings: bool) -> None:
        payload = [compact_properties(row, include_embeddings) for row in rows]
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (t:TextUnit {id: row.id})
            SET t += row
            """,
            rows=payload,
        )

    @staticmethod
    def _merge_entities(tx: Any, rows: List[Dict[str, Any]], include_embeddings: bool) -> None:
        for row in rows:
            props = compact_properties(row, include_embeddings)
            label = sanitize_label(str(row.get("entity_type", "Concept")))
            query = f"""
            MERGE (e:Entity:{label} {{id: $id}})
            SET e += $props
            """
            tx.run(query, id=row["id"], props=props)

    @staticmethod
    def _merge_communities(tx: Any, rows: List[Dict[str, Any]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (c:Community {id: row.id})
            SET c += row
            """,
            rows=rows,
        )

    @staticmethod
    def _link_document_text_units(tx: Any, text_units: List[Dict[str, Any]]) -> None:
        rows = [
            {"document_id": row["document_id"], "text_unit_id": row["id"]}
            for row in text_units
            if row.get("document_id") and row.get("id")
        ]
        if not rows:
            return
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.document_id})
            MATCH (t:TextUnit {id: row.text_unit_id})
            MERGE (d)-[:HAS_TEXT_UNIT]->(t)
            """,
            rows=rows,
        )

    @staticmethod
    def _link_document_entities(tx: Any, entities: List[Dict[str, Any]]) -> None:
        rows: List[Dict[str, Any]] = []
        for entity in entities:
            entity_id = entity.get("id")
            for document_id in entity.get("document_ids", []):
                rows.append({"document_id": document_id, "entity_id": entity_id})

        if not rows:
            return
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.document_id})
            MATCH (e:Entity {id: row.entity_id})
            MERGE (d)-[:CONTAINS_ENTITY]->(e)
            """,
            rows=rows,
        )

    @staticmethod
    def _link_text_unit_entities(tx: Any, entities: List[Dict[str, Any]]) -> None:
        rows: List[Dict[str, Any]] = []
        for entity in entities:
            entity_id = entity.get("id")
            for text_unit_id in entity.get("text_unit_ids", []):
                rows.append({"text_unit_id": text_unit_id, "entity_id": entity_id})

        if not rows:
            return
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (t:TextUnit {id: row.text_unit_id})
            MATCH (e:Entity {id: row.entity_id})
            MERGE (t)-[:MENTIONS]->(e)
            """,
            rows=rows,
        )

    @staticmethod
    def _link_community_entities(tx: Any, communities: List[Dict[str, Any]]) -> None:
        rows: List[Dict[str, Any]] = []
        for community in communities:
            community_id = community.get("id")
            for entity_id in community.get("entity_ids", []):
                rows.append({"community_id": community_id, "entity_id": entity_id})

        if not rows:
            return
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (c:Community {id: row.community_id})
            MATCH (e:Entity {id: row.entity_id})
            MERGE (c)-[:HAS_ENTITY]->(e)
            """,
            rows=rows,
        )

    @staticmethod
    def _merge_entity_relationships(
        tx: Any,
        rel_type: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        query = f"""
        UNWIND $rows AS row
        MATCH (source:Entity {{id: row.source}})
        MATCH (target:Entity {{id: row.target}})
        MERGE (source)-[r:{rel_type} {{id: row.id}}]->(target)
        SET r.weight = row.weight,
            r.relation_type = row.relation_type,
            r.evidence_text_unit_ids = row.evidence_text_unit_ids
        """
        tx.run(query, rows=rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 GraphRAG 输出导入 Neo4j 数据库")
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent / "graphrag_output"),
        help="GraphRAG 输出目录",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j 连接地址",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j 用户名",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j 密码",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j 数据库名",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="同时导入 embedding 向量属性",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="导入前清空已有的 Document/TextUnit/Entity/Community 节点",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.password:
        raise ValueError("请提供 Neo4j 密码，可通过 --password 或环境变量 NEO4J_PASSWORD 传入")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"未找到 GraphRAG 输出目录: {input_dir}")

    importer = Neo4jGraphImporter(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        include_embeddings=args.include_embeddings,
    )

    try:
        summary = importer.import_from_directory(input_dir=input_dir, clear_existing=args.clear)
    finally:
        importer.close()

    print("Neo4j 导入完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"数据库: {args.database}")
    print(f"输入目录: {input_dir}")


if __name__ == "__main__":
    main()
