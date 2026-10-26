from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCnYeXUUq_-pf4JBLtVmVzlbThgY0bZ_o8"
def get_llm():
    return ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0 )