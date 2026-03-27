import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    print("API_KEY not found in .env")
else:
    genai.configure(api_key=api_key)
    try:
        with open("models.txt", "w") as f:
            print("Listing available models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(m.name)
                    f.write(m.name + "\n")
        print("Models written to models.txt")
    except Exception as e:
        print(f"Error listing models: {e}")
