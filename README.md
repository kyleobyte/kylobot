# Kylobot
Kylobot is a grumpy teenage chatbot with simulated long-term memory powered by OpenAI's GPT-3.5-turbo and Pinecone. Kylobot can remember past conversations and provide contextual responses based on previous interactions.

## Getting Started
These instructions will help you set up and run Kylobot on your local machine.

## Prerequisites
Python 3.7+
An OpenAI API key: Sign up for an account at OpenAI
A Pinecone API key: Sign up for an account at Pinecone

## Installation
### Clone the repository:
bash
git clone https://github.com/your_username/Kylobot.git

### Change to the project directory:
bash
cd Kylobot

### Install the required dependencies:
pip install -r requirements.txt

## Configuration
Create a .env file in the project directory with the following content:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
DISCORD_TOKEN=your_discord_token
DISCORD_GUILD=your_discord_server
Replace your_openai_api_key and your_pinecone_api_key with your actual API keys.

## Usage
Import the handle_message function from kylobot.py in your desired application:
from kylobot import handle_message
Use the handle_message function to interact with Kylobot:
user_input = "What's your name?"
response = handle_message(user_input)
print(response)

##Acknowledgments
OpenAI for the GPT-3.5-turbo API
Pinecone for providing the vector search engine
Dave Shap for kicking my ass about sharing and documentation 
