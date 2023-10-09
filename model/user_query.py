from sentence_transformers import SentenceTransformer
import pinecone as pc
import sys
import initialize_db as init
import openai
from dotenv import load_dotenv


# Pinecone query
def query_pinecone(query, k, index, model):   
    query_em = model.encode(query).tolist()
    context = index.query(query_em, top_k=k, includeMetadata=True)
    return context


# Gpt prompt and response:
def gpt_responds(query, context):
    system_role="Answer the question as truthfully as possible using the provided context, " + \
    "and if the answer is not contained within the text and requires some latest information " + \
    "to be updated, print 'Not Sufficient context to answer query' \n" 
    context = context['matches'][0]['metadata']['content']
    user_input = context + '\n' + query +'\n'
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":system_role},
            {"role":"user","content":user_input}
        ]
    )
    return response

if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init.init_db()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = input("Ask fire-bot a question: ")
    index = pc.GRPCIndex("fireQuery")
    context = query_pinecone(query, 1, index, model)
    response = gpt_responds(query, context)
    print(response)

    

    