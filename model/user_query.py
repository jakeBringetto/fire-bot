from sentence_transformers import SentenceTransformer
import pinecone as pc
import sys
import initialize_db as init
import openai
from dotenv import load_dotenv
from dotenv import dotenv_values
import os


# Pinecone query
def query_pinecone(query, k, index, model):   
    query_em = model.encode(query).tolist()
    context = index.query(query_em, top_k=k, includeMetadata=True)
    final_context = ""
    for i in range(k):
        final_context += context['matches'][i]['metadata']['content'][0]
    return final_context


# Gpt prompt and response:
def gpt_responds(query, context):
    system_role="Answer the question as truthfully as possible using the provided context." 
    user_input = context + '\n' + query +'\n'
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":system_role},
            {"role":"user","content":user_input}
        ]
    )
    return response

def query(input_):
    config = dotenv_values("../.env")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    openai.api_key = config["OPENAI_API_KEY"]
    pc.init(api_key=config["PINECONE_API_KEY"], environment="gcp-starter")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = pc.Index("fire-query")
    context = query_pinecone(input_, 3, index, model)
    response = gpt_responds(input_, context)
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init.init_db()
    input_ = input("Ask fire-bot a question: ")
    response = query(input_)
    print(response)


    

    