from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
)

response = llm.invoke("How to be alpha")

print(response)
