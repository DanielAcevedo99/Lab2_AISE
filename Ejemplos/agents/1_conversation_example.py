import os
from dotenv import load_dotenv, find_dotenv
from autogen import ConversableAgent
import google.generativeai as genai

API_KEY = "AIzaSyC5L93vD_yXeVTKg__v1YbhnXALYA3LgAw"
genai.configure(api_key=API_KEY)

llm_config = { "model": "gemini-1.5-flash-latest", "api_key": API_KEY, "api_type": "google" }

cathy = ConversableAgent(
    name="cathy",
    system_message="Your name is Cathy and you are a stand-up comedian.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

joe = ConversableAgent(
    name="joe",
    system_message="Your name is Joe and you are a stand-up comedian. Start the next joke from the punchline of the previous joke.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

chat_result = joe.initiate_chat(
    recipient=cathy,
    message="I'm Joe. Cathy, let's keep the jokes rolling.",
    max_turns=2,
)

# Create a txt that traces the information printed in the terminal
with open("conversation.txt", "w", encoding="utf-8") as f:
    f.write(str(chat_result))

cathy.send(message="What's the last joke we said?", recipient=joe, request_reply=True)
