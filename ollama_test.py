print("--- SCRIPT STARTING ---")
from langchain_ollama import ChatOllama

try:
    print("Attempting to connect to Ollama...")
    llm = ChatOllama(model="llama3.1", temperature=0)
    response = llm.invoke("Say the word 'GOJO' if you are alive.")
    print(f"Llama Response: {response.content}")
except Exception as e:
    print(f"Error caught: {e}")
print("--- SCRIPT FINISHED ---")