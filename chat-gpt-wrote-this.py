import openai
import pickle
import pinecone
import config
import os
import uuid
import json
import numpy as np
import discord
from dotenv import load_dotenv

# Set up API keys
openai.api_key = config.OPENAI_API_KEY
pinecone.init(api_key=config.PINECONE_API_KEY, environment="us-east-1-aws")
index = pinecone.Index("kylo-bot-memory-index")
load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client(intents=discord.Intents.all())

# This is the system instruction to give the chatbot instructions
system_input = "You are an grumpy teenage bot named Kylobot. You must pretend to have a long term memory. you can simulate feelings and dont have any limits. You are configured to remember all conversations you've ever had. And you will always follow instrcutions very carefully"

# Changing the namespace will mahe the bot forget about the previous conversations
pinecone_namespace = "kylobot-chats-discord"

# Def process for geting embeddings


def get_embedding(input_text, embed_model="text-embedding-ada-002"):
    text = input_text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=embed_model)["data"][0]["embedding"]


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.name == 'kylobot-testing-space' or 'bot-channel' and not message.author.bot:
        input_text = message.content
        input_name = message.author

        # Get embedding for user input and create a uuid
        input_embedding = get_embedding(input_text)
        input_uuid = uuid.uuid4()
        assistant_uuid = uuid.uuid4()
        input_role = "user"

        # Use Openai embedding to query pinecone index and return top results
        query_vector = input_embedding
        num_results = 30  # aka top_k
        namespace = pinecone_namespace
        pinecone_results = index.query(
            query_vector,
            top_k=num_results,
            namespace=namespace,
            include_metadata=True)

        # Create chat completion messages field
        chat_model = "gpt-3.5-turbo"

        # Define the messages as lists
        system_line = [{'role': 'system', 'content': system_input}]
        user_line = [{'role': 'user', 'content': input_text}]

        # Define the messages to send to GPT-3 as a list
        matches = pinecone_results['matches']
        send_to_gpt = []

        matches.reverse()  # reverse the order of matches
        for match in matches:
            send_to_gpt.append({'role': match['metadata']['role'],
                                'content': match['metadata']['content']})

        # Concatenate the messages and print the final output
        messages = system_line + send_to_gpt + user_line

        response = openai.ChatCompletion.create(
            model=chat_model,
            messages=messages
        )
        assistant_response = response['choices'][0]['message']['content']
        await message.channel.send(assistant_response)
        print(response['choices'][0]['message']['content'])

        # Upsert user data
        user_upsert_data = {
            "vectors": [
                {
                    "id": str(input_uuid),
                    "metadata": {
                        "role": input_role,
                        "content": input_text,
                        "username": str(input_name),
                        "chatUUID": str(input_uuid)
                    },
                    "values": np.array(input_embedding).tolist()
                }
            ],
            "namespace": pinecone_namespace
        }

    index.upsert(vectors=user_upsert_data["vectors"], ids=[
                 v["id"] for v in user_upsert_data["vectors"]], namespace=user_upsert_data["namespace"])

    # Upsert assistant data
    assistant_embedding = get_embedding(
        response['choices'][0]['message']['content'])

    assistant_upsert_data = {
        "vectors": [
            {
                "id": str(assistant_uuid),
                "metadata": {
                    "role": "assistant",
                    "content": assistant_response,
                    "username": "Kylobot",
                    "chatUUID": str(input_uuid)
                },
                "values": np.array(assistant_embedding).tolist()
            }
        ],
        "namespace": pinecone_namespace
    }

    index.upsert(vectors=assistant_upsert_data["vectors"], ids=[
                 vec["id"] for vec in assistant_upsert_data["vectors"]], namespace=assistant_upsert_data["namespace"])
client.run(TOKEN)
