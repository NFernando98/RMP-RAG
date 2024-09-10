import json
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a Pinecone index
index_name = "rag"
dimension = 1536
try:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created.")
except Exception as e:
    print(f"Error creating index: {e}")

# Load the review data
with open("reviews.json") as f:
    data = json.load(f)

print("Review data loaded.")

# Initialize OpenAI client
client = OpenAI()

# Process and create embeddings
processed_data = []
for review in data["reviews"]:
    try:
        response = client.embeddings.create(
            input=review['review'], model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        print(f"Embedding dimension: {len(embedding)}")

        if len(embedding) == dimension:
            processed_data.append(
                {
                    "values": embedding,
                    "id": review["professor"],
                    "metadata": {
                        "review": review["review"],
                        "subject": review["subject"],
                        "stars": review["stars"],
                    }
                }
            )
        else:
            print(
                f"Skipping review for professor {review['professor']} due to dimension mismatch.")
    except Exception as e:
        print(
            f"Error creating embedding for professor {review['professor']}: {e}")

print("Embeddings created.")

# Upsert data into Pinecone
try:
    index = pc.Index(index_name)
    upsert_response = index.upsert(
        vectors=processed_data,
        namespace="ns1",
    )
    print(f"Upserted count: {upsert_response['upserted_count']}")
except Exception as e:
    print(f"Error upserting data into Pinecone: {e}")

# Print index statistics
try:
    stats = index.describe_index_stats()
    print(f"Index statistics: {stats}")
except Exception as e:
    print(f"Error describing index stats: {e}")
