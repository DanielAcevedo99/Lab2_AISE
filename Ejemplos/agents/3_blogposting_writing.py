import os
from dotenv import load_dotenv, find_dotenv
from autogen import AssistantAgent
import google.generativeai as genai

API_KEY = "AIzaSyC5L93vD_yXeVTKg__v1YbhnXALYA3LgAw"
genai.configure(api_key=API_KEY)
llm_config = { "model": "gemini-1.5-flash-latest", "api_key": API_KEY, "api_type": "google" }

# https://github.com/microsoft/autogen/issues/2860
#
# We have to Monkey patch [Your Python]/lib/python3.11/site-packages/autogen/agentchat/chat.py:119
# with the following code:
#
# _chat_carryover = chat_info.get("carryover", [])
# if isinstance(_chat_carryover, str):
#     _chat_carryover = [_chat_carryover]

# # Stringify everything in carryover
# processed_carryover = [str(item) for item in _chat_carryover]
# chat_info["carryover"] = processed_carryover
#

task = '''
  Write a concise but engaging blogpost about
  DeepLearning.AI. Make sure the blogpost is
  within 100 words.
'''

writer = AssistantAgent(
    name="Writer",
    system_message="You are a writer. You write engaging and concise "
        "blogpost (with title) on given topics. You must polish your "
        "writing based on the feedback you receive and give a refined "
        "version. Only return your final work without additional comments.",
    llm_config=llm_config,
)

critic = AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content.",
)

SEO_reviewer = AssistantAgent(
    name="SEO Reviewer",
    llm_config=llm_config,
    system_message="You are an SEO reviewer, known for "
        "your ability to optimize content for search engines, "
        "ensuring that it ranks well and attracts organic traffic. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

legal_reviewer = AssistantAgent(
    name="Legal Reviewer",
    llm_config=llm_config,
    system_message="You are a legal reviewer, known for "
        "your ability to ensure that content is legally compliant "
        "and free from any potential legal issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

ethics_reviewer = AssistantAgent(
    name="Ethics Reviewer",
    llm_config=llm_config,
    system_message="You are an ethics reviewer, known for "
        "your ability to ensure that content is ethically sound "
        "and free from any potential ethical issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

meta_reviewer = AssistantAgent(
    name="Meta Reviewer",
    llm_config=llm_config,
    system_message="You are a meta reviewer, you aggragate and review "
    "the work of other reviewers and give a final suggestion on the content.",
)

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content.
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

review_chats = [
  {
    "recipient": SEO_reviewer,
    "message": reflection_message,
    "summary_method": "reflection_with_llm",
    "summary_args": {
      "summary_prompt" :
        "Return review into as JSON object only: {'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
    },
    "max_turns": 1
  },
  {
    "recipient": legal_reviewer, "message": reflection_message,
    "summary_method": "reflection_with_llm",
    "summary_args": {
      "summary_prompt" :
        "Return review into as JSON object only: {'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
    },
    "max_turns": 1
  },
  {
    "recipient": ethics_reviewer, "message": reflection_message,
    "summary_method": "reflection_with_llm",
    "summary_args": {
      "summary_prompt" :
        "Return review into as JSON object only: {'reviewer': '', 'review': ''}.  Here Reviewer should be your role",
    },
    "max_turns": 1
  },
  {
    "recipient": meta_reviewer,
    "message": "Aggregrate feedback from all reviewers and give final suggestions on the writing.",
    "max_turns": 1
  },
]

critic.register_nested_chats(
    review_chats,
    trigger=writer,
)

res = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)

print(res.summary)
