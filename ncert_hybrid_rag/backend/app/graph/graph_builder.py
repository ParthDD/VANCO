import json
import re
from pathlib import Path
from neo4j import GraphDatabase
from ..core.config import (
    CHUNKS_FILE_PATH,
    GRAPH_FILE_PATH,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)


class GraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

        self.known_concepts = [
            "electric charge",
            "electric field",
            "field line",
            "superposition principle",
            "electric dipole",
            "dipole moment",
            "electric flux",
            "gauss law",
            "gauss's law",
            "coulomb law",
            "coulomb's law",
            "potential difference",
            "capacitance",
            "dielectric",
            "current",
            "resistance",
            "resistivity",
            "magnetic field",
            "electromagnetic wave",
            "test charge",
            "permittivity"
        ]

    def close(self):
        self.driver.close()

    def load_chunks(self, chunks_path: Path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def normalize_id(self, text: str):
        text = text.lower().strip()
        text = text.replace("'", "")
        text = re.sub(r"[^a-z0-9\s_]", "", text)
        text = re.sub(r"\s+", "_", text)
        return text

    def extract_section_titles(self, text: str):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections = []

        for line in lines:
            if re.match(r"^\d+(\.\d+)+\s+[A-Z].*", line):
                sections.append(line)
            elif re.match(r"^\d+\.\d+\s+[A-Z].*", line):
                sections.append(line)

        return sections

    def extract_formulas(self, text: str):
        formulas = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines:
            if "=" in line and len(line) < 140:
                formulas.append(line)

        return formulas[:5]

    def extract_concepts(self, text: str):
        found = []
        lower_text = text.lower()

        for concept in self.known_concepts:
            if concept in lower_text:
                found.append(concept)

        return sorted(list(set(found)))

    def infer_chapter(self, text: str):
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines[:6]:
            if len(line) < 80 and line.replace(" ", "").isalpha():
                if line.lower() not in {"physics", "points to ponder", "exercises", "contents"}:
                    return line

        return None

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (n:Page) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT formula_id IF NOT EXISTS FOR (n:Formula) REQUIRE n.id IS UNIQUE"
        ]

        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def merge_node(self, session, label, node_id, name, metadata=None):
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n.name = $name,
            n += $metadata
        """
        session.run(query, id=node_id, name=name, metadata=metadata or {})

    def merge_relationship(self, session, source_label, source_id, rel_type, target_label, target_id):
        query = f"""
        MATCH (a:{source_label} {{id: $source_id}})
        MATCH (b:{target_label} {{id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        session.run(query, source_id=source_id, target_id=target_id)

    def export_graph_snapshot(self, output_path: Path = GRAPH_FILE_PATH):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.driver.session() as session:
            node_result = session.run("""
                MATCH (n)
                RETURN labels(n) AS labels, n.id AS id, n.name AS name, properties(n) AS props
            """)
            edge_result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.id AS source, type(r) AS relation, b.id AS target
            """)

            nodes = []
            for record in node_result:
                labels = record["labels"]
                node_type = labels[0] if labels else "Node"
                props = dict(record["props"])
                nodes.append({
                    "id": record["id"],
                    "type": node_type,
                    "name": record["name"],
                    "metadata": props
                })

            edges = []
            for record in edge_result:
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"]
                })

            graph = {"nodes": nodes, "edges": edges}

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

        return output_path

    def build_graph(self, chunks_path: Path = CHUNKS_FILE_PATH):
        chunks = self.load_chunks(chunks_path)

        self.clear_graph()
        self.create_constraints()

        with self.driver.session() as session:
            document_id = "document_ncert_physics"
            self.merge_node(session, "Document", document_id, "NCERT Physics", {"source": str(chunks_path)})

            current_chapter_id = None

            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                page_number = chunk["page_number"]
                text = chunk["text"]

                page_id = f"page_{page_number}"
                chunk_node_id = f"chunk_{chunk_id}"

                self.merge_node(session, "Page", page_id, f"Page {page_number}", {"page_number": page_number})
                self.merge_node(
                    session,
                    "Chunk",
                    chunk_node_id,
                    chunk_id,
                    {"page_number": page_number, "text": text[:500]}
                )

                self.merge_relationship(session, "Document", document_id, "HAS_PAGE", "Page", page_id)
                self.merge_relationship(session, "Page", page_id, "HAS_CHUNK", "Chunk", chunk_node_id)

                chapter_name = self.infer_chapter(text)
                if chapter_name and len(chapter_name.split()) <= 8:
                    current_chapter_id = f"chapter_{self.normalize_id(chapter_name)}"
                    self.merge_node(session, "Chapter", current_chapter_id, chapter_name)
                    self.merge_relationship(session, "Document", document_id, "HAS_CHAPTER", "Chapter", current_chapter_id)

                if current_chapter_id:
                    self.merge_relationship(session, "Chapter", current_chapter_id, "ON_PAGE", "Page", page_id)
                    self.merge_relationship(session, "Chapter", current_chapter_id, "CONTAINS", "Chunk", chunk_node_id)

                sections = self.extract_section_titles(text)
                for section in sections:
                    section_id = f"section_{self.normalize_id(section)}"
                    self.merge_node(session, "Section", section_id, section)

                    if current_chapter_id:
                        self.merge_relationship(session, "Chapter", current_chapter_id, "HAS_SECTION", "Section", section_id)

                    self.merge_relationship(session, "Section", section_id, "CONTAINS", "Chunk", chunk_node_id)
                    self.merge_relationship(session, "Section", section_id, "ON_PAGE", "Page", page_id)

                concepts = self.extract_concepts(text)
                concept_ids = []

                for concept in concepts:
                    concept_id = f"concept_{self.normalize_id(concept)}"
                    concept_ids.append(concept_id)

                    self.merge_node(session, "Concept", concept_id, concept)
                    self.merge_relationship(session, "Chunk", chunk_node_id, "MENTIONS_CONCEPT", "Concept", concept_id)
                    self.merge_relationship(session, "Concept", concept_id, "ON_PAGE", "Page", page_id)

                    if current_chapter_id:
                        self.merge_relationship(session, "Chapter", current_chapter_id, "MENTIONS_CONCEPT", "Concept", concept_id)

                formulas = self.extract_formulas(text)
                for idx, formula in enumerate(formulas, start=1):
                    formula_id = f"formula_{chunk_id}_{idx}"
                    self.merge_node(session, "Formula", formula_id, formula)
                    self.merge_relationship(session, "Chunk", chunk_node_id, "HAS_FORMULA", "Formula", formula_id)
                    self.merge_relationship(session, "Formula", formula_id, "ON_PAGE", "Page", page_id)

                    for concept_id in concept_ids:
                        self.merge_relationship(session, "Concept", concept_id, "RELATED_TO", "Formula", formula_id)

                for i in range(len(concept_ids)):
                    for j in range(i + 1, len(concept_ids)):
                        c1 = concept_ids[i]
                        c2 = concept_ids[j]
                        self.merge_relationship(session, "Concept", c1, "RELATED_TO", "Concept", c2)
                        self.merge_relationship(session, "Concept", c2, "RELATED_TO", "Concept", c1)

    def build_and_save(self):
        self.build_graph()
        output_path = self.export_graph_snapshot()

        with self.driver.session() as session:
            node_result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
            """)

            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) AS count
            """)

            type_counts = {}
            total_nodes = 0

            for record in node_result:
                type_counts[record["label"]] = record["count"]
                total_nodes += record["count"]

            total_edges = rel_result.single()["count"]

        return {
            "graph_path": str(output_path),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_type_counts": type_counts
        }

    def query_graph(self, concept: str):
        concept = concept.lower().strip()

        with self.driver.session() as session:
            concept_query = """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS $concept
            RETURN c.id AS id, c.name AS name
            """
            concept_result = session.run(concept_query, concept=concept)

            matches = []
            matched_ids = []
            for record in concept_result:
                matches.append({
                    "id": record["id"],
                    "type": "Concept",
                    "name": record["name"],
                    "metadata": {}
                })
                matched_ids.append(record["id"])

            if not matched_ids:
                return {
                    "concept": concept,
                    "matches": [],
                    "connected_edges": [],
                    "connected_nodes": []
                }

            edge_query = """
            MATCH (a)-[r]->(b)
            WHERE a.id IN $matched_ids OR b.id IN $matched_ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target
            """
            node_query = """
            MATCH (n)
            WHERE n.id IN $node_ids
            RETURN labels(n)[0] AS type, n.id AS id, n.name AS name, properties(n) AS props
            """

            edge_result = session.run(edge_query, matched_ids=matched_ids)
            connected_edges = []
            node_ids = set(matched_ids)

            for record in edge_result:
                connected_edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"]
                })
                node_ids.add(record["source"])
                node_ids.add(record["target"])

            node_result = session.run(node_query, node_ids=list(node_ids))
            connected_nodes = []

            for record in node_result:
                connected_nodes.append({
                    "id": record["id"],
                    "type": record["type"],
                    "name": record["name"],
                    "metadata": dict(record["props"])
                })

        return {
            "concept": concept,
            "matches": matches,
            "connected_edges": connected_edges,
            "connected_nodes": connected_nodes
        }