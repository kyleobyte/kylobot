import openai
import pinecone
import os
import uuid
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
    # Move the index definition inside the function
    index = pinecone.Index("kylo-bot-memory")
    input_name = "User"
    # Get embedding for user input and create a uuid
    input_embedding = get_embedding(input_text)
    input_uuid = uuid.uuid4()
    assistant_uuid = uuid.uuid4()

    # Query Pinecone index for relevant conversation history
    query_vector = input_embedding
    num_results = 50
    pinecone_results = index.query(
        query_vector,
        top_k=num_results,
        namespace=pinecone_namespace,
        include_metadata=True)

    # Prepare message history for GPT-3
    system_line = [{'role': 'system', 'content': system_input}]
    user_line = [{'role': 'user', 'content': input_text}]
    matches = pinecone_results['matches']
    matches.reverse()
    message_history = [{'role': match['metadata']['role'],
                        'content': match['metadata']['content']} for match in matches]

    # Concatenate the messages and get GPT's response
    messages = system_line + message_history + user_line
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=messages)
    assistant_response = response['choices'][0]['message']['content']

    # Upsert user and assistant data
    upsert_data(str(input_uuid), "user", input_text,
                pinecone_namespace, input_embedding)
    assistant_embedding = get_embedding(assistant_response)
    upsert_data(str(assistant_uuid), "assistant", assistant_response,
                pinecone_namespace, assistant_embedding)

    return assistant_response
