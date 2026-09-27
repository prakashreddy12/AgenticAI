from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2:3b"
)

response = llm.invoke("Hi Kannayya")

print(response)
