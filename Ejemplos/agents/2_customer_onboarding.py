import os
from dotenv import load_dotenv, find_dotenv
from autogen import ConversableAgent, initiate_chats
import google.generativeai as genai

API_KEY = "AIzaSyC5L93vD_yXeVTKg__v1YbhnXALYA3LgAw"
genai.configure(api_key=API_KEY)

llm_config = { "model": "gemini-1.5-flash-latest", "api_key": API_KEY, "api_type": "google" }

def initiate_chats_with_json_parsing(chat_queue: list[dict[str, any]]) -> list:
    """
    Initiate chats with enhanced carryover processing to handle JSON.
    """
    finished_chats = []
    for chat_info in chat_queue:
        _chat_carryover = chat_info.get("carryover", [])
        if isinstance(_chat_carryover, str):
            _chat_carryover = [_chat_carryover]

        # Stringify everything in carryover
        processed_carryover = [str(item) for item in _chat_carryover]
        processed_carryover += [str(r.summary) for r in finished_chats]
        chat_info["carryover"] = processed_carryover

        # Initiate the chat
        chat_res = chat_info["sender"].initiate_chat(**chat_info)
        finished_chats.append(chat_res)
    return finished_chats

onboarding_personal_information_agent = ConversableAgent(
    name="Onboarding Personal Information Agent",
    system_message='''You are a helpful customer onboarding agent,
    you are here to help new customers get started with our product.
    Your job is to gather customer's name and location.
    Do not ask for other information. Return 'TERMINATE'
    when you have gathered all the information.''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

onboarding_topic_preference_agent = ConversableAgent(
    name="Onboarding Topic preference Agent",
    system_message='''You are a helpful customer onboarding agent,
    you are here to help new customers get started with our product.
    Your job is to gather customer's preferences on news topics.
    Do not ask for other information.
    Return 'TERMINATE' when you have gathered all the information.''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

customer_engagement_agent = ConversableAgent(
    name="Customer Engagement Agent",
    system_message='''You are a helpful customer service agent
    here to provide fun for the customer based on the user's
    personal information and topic preferences.
    This could include fun facts, jokes, or interesting stories.
    Make sure to make it engaging and fun!
    Return 'TERMINATE' when you are done.''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)

customer_proxy_agent = ConversableAgent(
    name="customer_proxy_agent",
    llm_config=False,
    code_execution_config=False,
    human_input_mode="ALWAYS",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)

chats = [
    {
        "sender": onboarding_personal_information_agent,
        "recipient": customer_proxy_agent,
        "message": "Hello, I'm here to help you get started with our product. "
                   "Could you tell me your name and location?",
        "summary_method": "reflection_with_llm",
        "summary_args": {
            "summary_prompt" : "Return the customer information "
                               "into as JSON object only: "
                               "{'name': '', 'location': ''}",
        },
        "max_turns": 2,
        "clear_history" : True
    },
    {
        "sender": onboarding_topic_preference_agent,
        "recipient": customer_proxy_agent,
        "message": "Great! Could you tell me what topics you are "
                   "interested in reading about?",
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
        "clear_history" : False
    },
    {
        "sender": customer_proxy_agent,
        "recipient": customer_engagement_agent,
        "message": "Let's find something fun to read.",
        "max_turns": 1,
        "summary_method": "reflection_with_llm",
    },
]

# Initiate chats and get the results
chat_results = initiate_chats_with_json_parsing(chats)

# Open a file to write the trace information
with open("customer_info.txt", "w", encoding="utf-8") as f:
    for chat_result in chat_results:
        f.write(f"Summary: {chat_result.summary}\n")
        f.write("\n")

    f.write("\nCosts:\n")
    for chat_result in chat_results:
        f.write(f"{chat_result.cost}\n")
        f.write("\n")

print("Trace information saved to trace_info.txt")
