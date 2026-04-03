from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
def main():
    print("Hello from dev!")


if __name__ == "__main__":
    main()
