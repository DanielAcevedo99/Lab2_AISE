from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv, find_dotenv
import autogen
from typing_extensions import Annotated
import google.generativeai as genai

API_KEY = "AIzaSyC5L93vD_yXeVTKg__v1YbhnXALYA3LgAw"
genai.configure(api_key=API_KEY)

llm_config = {
  "model": "gemini-1.5-flash-latest",
  "api_key": API_KEY,
  "api_type": "google"
}


task = "Write a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-07-18."

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="Give the task, and send instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)

planner = autogen.ConversableAgent(
    name="Planner",
    system_message="Given a task, please determine "
    "what information is needed to complete the task. "
    "Please note that the all the stock information will be retrieved using the researcher. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps. If a step fails, try to workaround",
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps",
    llm_config=llm_config,
)

researcher = autogen.ConversableAgent(
    name="Researcher",
    system_message="You researched the web for the topic you are provided, "
        "and the results you got are: "
        "Nvidia stock prices in the past month: "
        "18 Jun 135.88 USD, "
        "24 Jun 118.11 USD, "
        "1 Jul 124.30 USD, "
        "10 Jul 134.91 USD, "
        "17 Jul 117.97 USD",
    llm_config=llm_config,
)

writer = autogen.ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer."
    "Please write blogs in markdown format (with relevant titles)"
    " and put the content in pseudo ```md``` code block. "
    "You take feedback from the admin and refine your blog.",
    description="Writer."
    "Write blogs based on the google search results and take "
    "feedback from the admin to refine the blog."
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer, planner],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [researcher, writer, planner],
        researcher: [planner],
        writer: [planner],
        planner: [user_proxy, researcher, writer],
    },
    speaker_transitions_type="allowed",
)

manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)
