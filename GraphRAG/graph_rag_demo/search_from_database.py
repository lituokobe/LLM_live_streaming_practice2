# TODO: connect to database
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from Live_Streaming_practice2.GraphRAG.graph_rag_demo.llm import llm

graph_db = Neo4jGraph(
    url='bolt://localhost:7687',
    # url='neo4j://127.0.0.1:7687',
    username='neo4j',
    password='Lituo1988!',
    database='neo4j',
    enhanced_schema=True,
)

# TODO: Define a prompt template
cypher_template = PromptTemplate(
    input_variables=["schema", "question"],
    template=(
    "Based on the following Neo4j graph schema, generate a Cypher query."
    "Nodes must be matched using the id attribute as a priority; do not use alias."
    "If multiple entities need to be queried, each alias must be unique."
    "{schema}\n"
    "Question: {question}\n"
    "Example: MATCH (c:Country {{id: 'Japan'}})-[r]-(x) RETURN c, r, x LIMIT 5 — Please generate a Cypher query based on this example to return the node, the relationship, and the connected node. Do not return only p."
    "Please output only the Cypher query."
    ),
)

qa_prompt = PromptTemplate(
    input_variables=["question", "result"],
    template=(
        "Below is the answer to the user's question {question},"
        "based on the results retrieved from a Neo4j query. Do not fabricate any information."
    ),
)

# #TODO: Create a search object - #1
# # LLM is used twice: generate Cypher query and summarize the query result
# runnable = GraphCypherQAChain.from_llm(
#     llm = llm,
#     # cypher_llm = llm, # if you need to separate the llm for search and answer
#     # qa_llm = llm,
#     graph = graph_db,
#     verbose = True,
#     cypher_prompt = cypher_template,
#     qa_prompt = qa_prompt,
#     validate_cypher = True,
#     allow_dangerous_requests = True,
#     return_intermediate_steps = True, # You can see the SQL query generated
# )

#TODO: Create a search object - #2
runnable = GraphCypherQAChain.from_llm(
    llm = llm,
    graph = graph_db,
    verbose = True,
    cypher_prompt = cypher_template,
    validate_cypher = True,
    allow_dangerous_requests = True,
    return_intermediate_steps = True,
    use_function_response = True,
    top_k = 30
)

resp = runnable.invoke({"query": "台湾和中华民国的关系？"})
print(resp)
