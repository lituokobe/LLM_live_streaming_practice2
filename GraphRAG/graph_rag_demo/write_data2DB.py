import asyncio
import hashlib
import unicodedata
from typing import List
import re
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

from Live_Streaming_practice2.GraphRAG.graph_rag_demo.llm import llm
from Live_Streaming_practice2.GraphRAG.graph_rag_demo.obtain_data import loaded_docs

# TODO: connect to database
graph_db = Neo4jGraph(
    url='bolt://localhost:7687',
    # url='neo4j://127.0.0.1:7687',
    username='neo4j',
    password='Lituo1988!',
    database='neo4j',
    enhanced_schema=True,
)

# TODO: Get database's graph structure
# schema = graph_db.schema
# print(schema)

# TODO: the class of Neo4Graph can directly run SLQ query
# resp = graph_db.query("MATCH (n) DETACH DELETE n")
# print(resp)

# TODO: Build constraint to let the node name is unique
graph_db.query("""
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:`__Entity__`) REQUIRE e.id IS UNIQUE
""")

# TODO Regularize and stabilize ID
def stable_doc_id(meta: dict) -> str:
    """
    Use WikipediaLoader's metadata['source'] URL as a stabilizer
    if it doesn't exist, back to (title + summary)'s hash
    :param meta: a dictionary containing metadata about a document
    :return: Converts the base string to bytes and hashes it using MD5.
    Returns the hash as a hexadecimal string — this becomes the document ID
    The returned string is a unique and stable identifier for a document.
    """
    base = meta.get("source") or (meta.get("title","") + "|" + meta.get("summary",""))
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def attach_doc_ids(docs: List[Document]) -> List[Document]:
    for d in docs:
        d.metadata = dict(d.metadata)  # copy, avoid in-place side effect
        d.metadata["id"] = stable_doc_id(d.metadata)
        # also uniform metadata, for later query
        d.metadata.setdefault("source_type", "wikipedia")
        d.metadata.setdefault("title", d.metadata.get("title",""))
        d.metadata.setdefault("summary", d.metadata.get("summary",""))
    return docs

def normalize_id(s: str) -> str:
    """
    Uniform node ID: full-width half-width, whitespace, case, etc.
    :param s: a string
    :return: a string with uniformed ID
    """
    if not isinstance(s, str):
        return s
    # Normalization Form KC, decompose characters and replace compatible characters to their standard form
    # "Ｈｅｌｌｏ　Ｗｏｒｌｄ！"  ---> "Hello World!"
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# TODO: extract entity and relationship from the graph
allowed_nodes=["Person", "War", "Event", "Department", "Organization", "Country", "City", "Position"]

allowed_relationships=[
        ("Person", "LEADER_OF", "Country"),
        ("Country", "PARTICIPANT_OF", "War"),
        ("Person", "FOUNDER_OF", "Organization"),
        ("Person", "HEAD_OF", "Organization"),
        ("Person", "MEMBER_OF", "Department"),
        ("Person", "PARTICIPANT", "Event"),
        ("Person", "LEADER_OF", "City"),
        ("Country", "ALLIED_WITH", "Country"),
        ("Country", "FOUGHT_AGAINST", "Country"),
        ("Country", "RECEIVE_AID_FROM", "Country"),
        ("Person", "HAD_WAR_IN", "Country"),
        ("War", "HAPPENED_IN", "Country"),
        ("Person", "HAS_POSITION", "Position"),
        ("Event", "HAPPENED_DURING", "War"),
        ("Person", "HAS_PRODUCT_OF", "Product"),
        ("Country", "WINNER_OF", "War")
]

node_properties=["alias", "dob", "founded_year", "citizenship", "ticker", "city", "country", "department", "position"]

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    node_properties=node_properties,
    strict_mode=True # If True, when input has more entities than allowed_nodes, those entities won't be extracted
)

async def extract_graph(docs: List[Document]) -> List[GraphDocument]:
    """
    Extract the entities. Convert documents to graph structure
    :param docs: a list of documents
    :return: a list of graph documents
    """
    # Every element is an object of GraphDocument
    graphs = await llm_transformer.aconvert_to_graph_documents(docs)
    for gd in graphs:
        for n in gd.nodes:
            n.id = normalize_id(n.id)
        for r in gd.relationships:
            r.source.id = normalize_id(r.source.id)
            r.target.id = normalize_id(r.target.id)
    return graphs

#TODO: Write to Neo4j database
def upsert_to_neo4j(graph_documents):
    """
    Insert the graph documents into the graph database
    :param graph_documents: a list of graph documents
    """
    graph_db.add_graph_documents(
        graph_documents,
        baseEntityLabel=True, # Add __Entity__ secondary label, for easy indexing
        include_source=True # Include node from source document
    )

# TODO update from existing Wikipedia documents
async def ingest_wikipedia_docs(docs: List[Document]):
    docs = attach_doc_ids(docs)
    gdocs = await extract_graph(docs)
    print(gdocs)
    upsert_to_neo4j(gdocs)

# ==== 使用举例 ====
# from langchain_community.document_loaders import WikipediaLoader
# docs = WikipediaLoader(query="马云", lang="zh", load_max_docs=5).load()
asyncio.run(ingest_wikipedia_docs(loaded_docs))

