"""SambaNova on OpenAI Client Sample"""
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from toolhouse import Toolhouse # Import the Toolhouse SDK

#  Make sure to set up the .env file according to the .env.example file.
load_dotenv()

TH_API_KEY = os.getenv("TOOLHOUSE_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

client = OpenAI(
    api_key=SAMBANOVA_API_KEY,
    base_url="https://api.sambanova.ai/v1/"
)

# Initialize Toolhouse with the OpenAI provider
th = Toolhouse(api_key=TH_API_KEY, provider="openai")

messages: List = [{
    "role": "user",
    "content":
        "Scrape data from https://docs.sambanova.ai/cloud/docs/get-started/overview and tell me what SambaNova Cloud is."
}]

response = client.chat.completions.create(
    model='Meta-Llama-3.3-70B-Instruct',
    messages=messages,
    # Retrieve tools installed from Toolhouse
    tools=th.get_tools()
)

# Run the tools using the Toolhouse client with the created message
messages += th.run_tools(response)

response = client.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=messages,
            # Retrieve tools installed from Toolhouse
            tools=th.get_tools()
        )
print(response.choices[0].message.content)
