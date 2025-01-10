import numpy as np
from pymilvus import (
    MilvusClient,
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Vector parameters
_BATCH = 1000
_ROWS = 3139
_DIM = 256
_INDEX_FILE_SIZE = 32  # max file size of stored index
_NQ = 5

# Index parameters
_METRIC_TYPE = 'COSINE'
_INDEX_TYPE = 'HNSW'
_NLIST = 1024
_NPROBE = 16
_TOPK = 10
_EFC = 360
_EF = 16
_M = 8


def main():
    client = MilvusClient("./milvus_demo.db")
    client.create_collection(
        collection_name="demo_collection",
        dimension=384  # The vectors we will use in this demo has 384 dimensions
    )

    # Text strings to search from.
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    # For illustration, here we use fake vectors with random numbers (384 dimension).

    vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
    data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
    res = client.insert(
        collection_name="demo_collection",
        data=data
    )

    # This will exclude any text in "history" subject despite close to the query vector.
    res = client.search(
        collection_name="demo_collection",
        data=[vectors[0]],
        filter="subject == 'history'",
        limit=2,
        output_fields=["text", "subject"],
    )
    print(res)

    # a query that retrieves all entities matching filter expressions.
    res = client.query(
        collection_name="demo_collection",
        filter="subject == 'history'",
        output_fields=["text", "subject"],
    )
    print(res)

    # delete
    res = client.delete(
        collection_name="demo_collection",
        filter="subject == 'history'",
    )
    print(res)


if __name__ == '__main__':
    main()
