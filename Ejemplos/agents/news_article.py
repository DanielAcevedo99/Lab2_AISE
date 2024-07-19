import requests
from datetime import datetime
import autogen
import google.generativeai as genai

NEWS_API_KEY = "91b9c7e6fab94e12a0027fde6ed790e8"
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

response = requests.get(url)
articles = response.json().get('articles', [])

news_info = []
for article in articles:
    news_info.append({
        "title": article.get("title"),
        "description": article.get("description"),
        "content": article.get("content"),
        "url": article.get("url"),
        "publishedAt": article.get("publishedAt"),
        "source": article.get("source", {}).get("name")
    })

for news in news_info:
    print(news)

API_KEY_GENAI = "AIzaSyC5L93vD_yXeVTKg__v1YbhnXALYA3LgAw"
genai.configure(api_key=API_KEY_GENAI)

llm_config = {
    "model": "gemini-1.5-flash-latest",
    "api_key": API_KEY_GENAI,
    "api_type": "google"
}

task = "Write an article summarizing the following news information:\n\n"
for news in news_info:
    task += f"Title: {news['title']}\nDescription: {news['description']}\nContent: {news['content']}\nURL: {news['url']}\nPublished At: {news['publishedAt']}\nSource: {news['source']}\n\n"

# Define agents
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
                   "Check the progress and instruct the remaining steps. If a step fails, try to workaround.",
    description="Planner. Given a task, determine what "
                "information is needed to complete the task. "
                "Check the progress and instruct the remaining steps.",
    llm_config=llm_config,
)

writer = autogen.ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer. Please write articles in markdown format (with relevant titles) "
                   "based on the provided news information. "
                   "Take feedback from the admin and refine your article.",
    description="Writer. Write articles based on the provided news information "
                "and take feedback from the admin to refine the article."
)

critic = autogen.ConversableAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of the writer and provide constructive "
                   "feedback to help improve the quality of the content.",
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, writer, planner, critic],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [writer, planner],
        writer: [planner, critic],
        planner: [user_proxy, writer, critic],
        critic: [user_proxy, writer, planner],
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

print(groupchat_result.summary)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_file = f"trace_info_{timestamp}.txt"

with open(trace_file, "w", encoding="utf-8") as f:
    f.write("Task: \n")
    f.write(f"{task}\n\n")

    for agent in groupchat.agents:
        f.write(f"Agent: {agent.name}\n")
        f.write("Messages:\n")
        for message in agent.messages:
            f.write(f"{message}\n")
        f.write("\n")

print(f"Trace information saved to {trace_file}")
