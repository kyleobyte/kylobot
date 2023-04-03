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
    # Initialize the Pinecone index
    index = pinecone.Index("kylo-bot-memory")
    input_name = "User"

    # Get the input text's embedding and a UUID for the message
    input_embedding = get_embedding(input_text)
    input_uuid = uuid.uuid4()

    # Set up the query vector and parameters for Pinecone
    query_vector = input_embedding
    num_results = 20
    namespace = pinecone_namespace
    
    # Query Pinecone for the top_k most similar conversation fragments
    pinecone_results = index.query(query_vector, top_k=num_results, namespace=namespace, include_metadata=True)

    # Set up GPT-3.5-turbo
    chat_model = "gpt-3.5-turbo"

    # Define the system and user message lines
    system_line = [{'role': 'system', 'content': system_input}]
    user_line = [{'role': 'user', 'content': input_text}]

    matches = pinecone_results['matches']
    send_to_gpt = []

    # Set similarity score threshold and time-based factor
    score_threshold = 0.8
    time_based_factor = 0.01

    # Filter matches based on the similarity score threshold
    filtered_matches = [match for match in matches if match['score'] >= score_threshold]

    # Calculate the current time
    current_time = time.time()

    # Calculate the weighted score for each match considering the time-based factor
    for match in filtered_matches:
        match['weighted_score'] = match['score'] * (1 + time_based_factor * (current_time - uuid.UUID(match['metadata']['chatUUID']).time_low))
    
    # Sort the matches by their weighted scores
    sorted_matches = sorted(filtered_matches, key=lambda x: x['weighted_score'], reverse=True)

    # Create the list of past messages to send to GPT-3.5-turbo
    for match in sorted_matches:
        send_to_gpt.append({'role': match['metadata']['role'], 'content': match['metadata']['content']})

    # Combine system, past messages, and user message into one list
    messages = system_line + send_to_gpt[-num_results:] + user_line

    # Send the messages to GPT-3.5-turbo for generating a response
    response = openai.ChatCompletion.create(model=chat_model, messages=messages)
    assistant_response = response['choices'][0]['message']['content']

    # Upsert the user's message data into Pinecone
    upsert_conversation_data(input_uuid, input_embedding, input_text, input_name, input_role="user")
    
    # Upsert Kylobot's response data into Pinecone
    assistant_embedding = get_embedding(assistant_response)
    upsert_conversation_data(input_uuid, assistant_embedding, assistant_response, "Kylobot", input_role="assistant")

    return assistant_response
