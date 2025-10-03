import asyncio
import os

import pandas as pd
import tiktoken

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
import logging

from dotenv import load_dotenv
load_dotenv(dotenv_path="./test_graphrag/.env")
api_key = os.getenv("GRAPHRAG_API_KEY")

# 配置日志输出到控制台和文件
logging.basicConfig(
    level=logging.DEBUG,  # 设置为最低级别以捕获所有信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graphrag_debug.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

# 确保 GraphRAG 相关模块启用调试
logger = logging.getLogger("graphrag")
logger.setLevel(logging.DEBUG)

llm_model = 'gpt-4o-mini'

# 创建了大模型连接对象
config = LanguageModelConfig(
    api_key=api_key,
    api_base='https://api.openai.com/v1',
    type=ModelType.OpenAIChat,
    model=llm_model,
    max_retries=3,
)
model = ModelManager().get_or_create_chat_model(
    name="global_search",
    model_type=ModelType.OpenAIChat,
    config=config,
)

token_encoder = tiktoken.encoding_for_model(llm_model)

# parquet files generated from indexing pipeline
INPUT_DIR = "./test_graphrag/output"
COMMUNITY_TABLE = "communities"
COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2

community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")

communities = read_indexer_communities(community_df, report_df)
reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)

print(f"Total report count: {len(report_df)}")
print(
    f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
)

print(report_df.head())
print(entity_df.head())
print(community_df.head())



context_builder = GlobalCommunityContext(
    community_reports=reports,
    communities=communities,
    entities=entities,  # default to None if you don't want to use community weights for ranking
    # token_encoder=token_encoder,
)

context_builder_params = {
    "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}

search_engine = GlobalSearch(
    model=model,
    context_builder=context_builder,
    # token_encoder=token_encoder,
    max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
    json_mode=True,  # set this to False if your LLM model does not support JSON mode.
    context_builder_params=context_builder_params,
    concurrent_coroutines=32,
    response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
)

async def run_search():
    result = await search_engine.search("清朝和欧洲国家发生了哪些战争？")

    print(result.response)


if __name__ == '__main__':
    asyncio.run(run_search())