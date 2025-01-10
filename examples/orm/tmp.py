import random

from pymilvus import (
    MilvusClient,
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

COLLECTION_NAME = 'demo'
DIMENSION = 2048
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = '19530'

# Const names
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

    # Remove any previous collections with the same name
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # Create schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    # Add fields to schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="filepath", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="image_embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)

    # Prepare index params
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="image_embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    # Create a collection with the index loaded simultaneously
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    res = client.get_load_state(
        collection_name=COLLECTION_NAME
    )
    print(res)

    client.drop_collection(collection_name=COLLECTION_NAME)


if __name__ == '__main__':
    main()
