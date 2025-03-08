"""
Check if the OpenAI API key is loaded correctly
"""
import os
from dotenv import load_dotenv

# Explicitly load from the .env file in the current directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
print(f"Loading .env file from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

# Check the API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY is not set in the environment")
elif api_key == "your_openai_api_key":
    print("ERROR: OPENAI_API_KEY is set to the placeholder value 'your_openai_api_key'")
else:
    # Safely display part of the API key
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"SUCCESS: OPENAI_API_KEY is set to a valid value starting with {masked_key}")

print("\nAll environment variables:")
for key, value in os.environ.items():
    if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
        # Mask sensitive values
        masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        print(f"{key}={masked_value}")
    elif key.startswith('OPENAI_') or key.startswith('API_'):
        print(f"{key}={value}")

# Try importing the OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    print("\nSuccessfully created OpenAI client")
except Exception as e:
    print(f"\nError creating OpenAI client: {e}") 