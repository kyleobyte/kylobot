import discord
import os
from kylobot import handle_message
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client(intents=discord.Intents.all())


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.name in ('kylobot-testing-space', 'bot-channel') and not message.author.bot:
        input_text = message.content
        assistant_response = handle_message(input_text)
        await message.channel.send(assistant_response)

client.run(TOKEN)
