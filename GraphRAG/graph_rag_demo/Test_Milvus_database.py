from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://127.0.0.1:19530",
    # token="root:Milvus",
    user='root',
    password='Milvus',
    # db_name='default', # If you don't specify the db name, there will be a default db named "default"
)

# print(client.list_databases())
# print(client.list_users())
# print(client.list_collections())

# Create database
client.create_database(
    db_name="my_database",
    properties={
        "database.max.collections": 10
    }
)
print(client.list_databases())
# check database info
print(client.describe_database(
    db_name="my_database"
))
# # 使用数据库（切换数据库）
# client.use_database(
#     db_name="my_database"
# )
# # 删除数据库
# client.drop_database(
#     db_name="my_database"
# )
# print(client.list_databases())