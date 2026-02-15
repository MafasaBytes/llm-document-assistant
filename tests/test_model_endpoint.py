from langchain_ollama import OllamaLLM

def get_llm():
    llm = OllamaLLM(
        model="phi3:latest", 
        base_url="http://127.0.0.1:11434",
        temperature=0.5,
    )
    return llm

llm = get_llm()
response = llm.invoke("hi")
print(response)
