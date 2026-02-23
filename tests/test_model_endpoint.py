from langchain_ollama import OllamaLLM

def get_llm():
    """Test model endpoint..."""
    llm = OllamaLLM(
        model="mistral:7b", 
        base_url="http://127.0.0.1:11434",
        temperature=0.5,
    )
    return llm

llm = get_llm()
response = llm.invoke("hi")
print(response)