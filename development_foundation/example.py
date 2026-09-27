""""Make one llm call and read the response"""

import sys
import ollama

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "gpt-oss:120b-cloud"

request = input("Enter an IT request: ")
response = ollama.chat(model = MODEL, messages=
[
    {"role":"user", "message":f'Classify the following IT request:{request}'},
],
)

print("response: ", response["message"]["content"])