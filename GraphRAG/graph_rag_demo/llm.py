import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4.1-nano",
                 temperature=0.6,
                 api_key=OPENAI_API_KEY,
                 base_url='https://api.openai.com/v1')

