"""Setup Pinecone index."""
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


def setup_pinecone():
    """Create Pinecone index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rc-rag-multimodal")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    dimension = int(os.getenv("OPENAI_EMBEDDING_DIMS", 3072))
    metric = os.getenv("PINECONE_METRIC", "cosine")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in existing_indexes]
    
    if index_name not in index_names:
        print(f"Creating index '{index_name}'...")
        
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region=environment.split('-')[0:2]  # e.g., 'us-east-1'
            )
        )
        
        print(f"✓ Index '{index_name}' created successfully!")
    else:
        print(f"✓ Index '{index_name}' already exists")
    
    # Get index stats
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"\nIndex Stats:")
    print(f"  Dimension: {stats.dimension}")
    print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else ['default']}")


if __name__ == "__main__":
    setup_pinecone()
