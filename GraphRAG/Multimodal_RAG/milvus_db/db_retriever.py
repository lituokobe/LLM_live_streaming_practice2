from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker


class MilvusRetriever:
    def __init__(self, collection_name:str, milvus_client: MilvusClient, top_k: int=5):
        self.collection_name = collection_name
        self.client:MilvusClient = milvus_client
        self.top_k = top_k

    def dense_search(self, query_embedding, limit = 5):
        """
        Dense vector search
        :param query_embedding: embedded content
        :param limit:
        :return:
        """
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="dense",  # dense vector has image and text
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params=search_params,
        )
        return res[0]
    def sparse_search(self, query, limit = 5):
        """
        Sparse vector search. Search the full context
        :param query:  search the key word
        :param limit:
        :return:
        """
        return self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",  # only retrieve text
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params={"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}},
        )[0]
    def hybrid_search(
            self,
            query_dense_embedding,
            query_sparse_embedding,
            sparse_weight=1.0,
            dense_weight=1.0,
            limit=10,
    ):
        dense_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense", dense_search_params, limit=limit
        )
        sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse", sparse_search_params, limit=limit
        )
        # 重排算法
        rerank = WeightedRanker(sparse_weight, dense_weight)
        return self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_req, dense_req],
            ranker=rerank,  # 重排算法
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title']
        )[0]