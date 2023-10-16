import process_data as pd
from sentence_transformers import SentenceTransformer
import pinecone as pc
from dotenv import load_dotenv
from dotenv import dotenv_values


# Create pinecone index
def create_pinecone():
    # config = dotenv_values("../.env")
    pc.create_index("fire-query", dimension=384)
    pc.describe_index("fire-query")

# Insert 
def insert_pincone(data, model):
    index = pc.Index("fire-query")
    upserted_data = []
    i=0
    for item in data:
        id_ = index.describe_index_stats()['total_vector_count']
        upserted_data.append(
            (
                str(id_+i),
                model.encode(item).tolist(),
                {
                    'content': item
                }
            )
        )
        i+=1
        index.upsert(vectors=upserted_data)

# initialize database
def init_db():
    data = pd.csv_to_array()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # create_pinecone()
    insert_pincone(data, model)

if __name__ == "__main__":
    config = dotenv_values("../.env")
    pc.init(api_key=config["PINECONE_API_KEY"], environment="gcp-starter")
    init_db()
