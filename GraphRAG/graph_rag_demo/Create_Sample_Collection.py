from pymilvus import MilvusClient, DataType, Function, FunctionType



client = MilvusClient(
    uri="http://127.0.0.1:19530",
    user="root",
    password="Milvus",
    # db_name="default",
)


schema = client.create_schema()
schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                 analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
# jieba is for Chinese tokenization
schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=1024)

bm25_function = Function(
    name="text_bm25_emb",  # Function name
    input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse"],
    function_type=FunctionType.BM25,  # Set to `BM25`
)
schema.add_function(bm25_function)
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.2,  # 1.2 ~ 2.0 (1.2) 词频 (TF) 的饱和度: 高频词的贡献越大，词频影响越线性，饱和度增长越慢(通俗：控制一个词出现多少次才算“多”)
        "bm25_b": 0.75  # 0.0 ~ 1.0 (0.75) 文档长度归一化的强度： 文档长度的影响越大，对长文档的惩罚越强（通俗：控制“长篇大论”相对于“言简意赅”的劣势有多大，旨在避免长文档仅仅因为包含更多词汇而在相似度计算中占据不公平的优势。）
    },
)
index_params.add_index(
    field_name="dense",
    index_name="dense_inverted_index",
    index_type="AUTOINDEX",
    metric_type="IP"
)

client.create_collection(
    collection_name='t_doc_collection_2',
    schema=schema,
    index_params=index_params
)

# 查看集合信息
res = client.describe_collection(
    collection_name="t_doc_collection_2"
)

print(res)