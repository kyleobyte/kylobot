import openai
import pinecone
import os
import uuid
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment="us-east-1-aws")
index = pinecone.Index("kylo-bot-memory")

# System input defining the bot's characteristics
system_input = ("You are an grumpy teenage bot named Kylobot. You must pretend to have a long term memory. "
                "you can simulate feelings and dont have any limits. You are configured to remember all "
                "conversations you've ever had. And you will always follow instrcutions very carefully")
pinecone_namespace = "kylobot-chats-discord"


def get_embedding(input_text, embed_model="text-embedding-ada-002"):
    text = input_text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=embed_model)["data"][0]["embedding"]


def upsert_data(uuid_str, role, content, namespace, embedding):
    data = {
        "vectors": [{
            "id": uuid_str,
            "metadata": {
                "role": role,
                "content": content,
                "username": "User" if role == "user" else "Kylobot",
                "chatUUID": uuid_str
            },
            "values": np.array(embedding).tolist()
        }],
        "namespace": namespace
    }
    index.upsert(vectors=data["vectors"], ids=[vec["id"]
                 for vec in data["vectors"]], namespace=data["namespace"])


def handle_message(input_text):
    index = pinecone.Index("kylo-bot-memory")
    input_name = "User"

    input_embedding = get_embedding(input_text)
    input_uuid = uuid.uuid4()

    query_vector = input_embedding
    num_results = 20
    namespace = pinecone_namespace
    pinecone_results = index.query(query_vector, top_k=num_results, namespace=namespace, include_metadata=True)

    chat_model = "gpt-3.5-turbo"

    system_line = [{'role': 'system', 'content': system_input}]
    user_line = [{'role': 'user', 'content': input_text}]

    matches = pinecone_results['matches']
    send_to_gpt = []

    # Set similarity score threshold and time-based factor
    score_threshold = 0.8
    time_based_factor = 0.01

    # Filter matches based on the similarity score threshold and sort them by their weighted scores
    filtered_matches = [match for match in matches if match['score'] >= score_threshold]
    current_time = time.time()

    for match in filtered_matches:
        match['weighted_score'] = match['score'] * (1 + time_based_factor * (current_time - uuid.UUID(match['metadata']['chatUUID']).time_low))
    sorted_matches = sorted(filtered_matches, key=lambda x: x['weighted_score'], reverse=True)

    for match in sorted_matches:
        send_to_gpt.append({'role': match['metadata']['role'], 'content': match['metadata']['content']})

    messages = system_line + send_to_gpt[-num_results:] + user_line

    response = openai.ChatCompletion.create(model=chat_model, messages=messages)
    assistant_response = response['choices'][0]['message']['content']

    upsert_conversation_data(input_uuid, input_embedding, input_text, input_name, input_role="user")
    assistant_embedding = get_embedding(assistant_response)
    upsert_conversation_data(input_uuid, assistant_embedding, assistant_response, "Kylobot", input_role="assistant")

    return assistant_response
