"""SambaNova on OpenAI Sample"""
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from toolhouse import Toolhouse # Import the Toolhouse SDK
from toolhouse.models.Stream import ToolhouseStreamStorage

#  Make sure to set up the .env file according to the .env.example file.
load_dotenv()

TH_API_KEY = os.getenv("TOOLHOUSE_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

client = OpenAI(
    api_key=SAMBANOVA_API_KEY,
    base_url="https://api.sambanova.ai/v1/",
)

# Initialize Toolhouse with the OpenAI provider
th = Toolhouse(api_key=TH_API_KEY, provider="openai")

messages: List = [{
    "role": "user",
    "content":
        "Generate code to calculate the Fibonacci sequence to 100." "Execute it and give me the result"
}]

stream = client.chat.completions.create(
    model='Meta-Llama-3.1-405B-Instruct',
    messages=messages,
    # Retrieve tools installed from Toolhouse
    tools=th.get_tools(),
    stream=True
)

# Use the stream and save blocks
stream_storage = ToolhouseStreamStorage()
for block in stream:  # pylint: disable=E1133
    print(block)
    stream_storage.add(block)

# Run the tools using the Toolhouse client with the created message
messages += th.run_tools(stream_storage)

response = client.chat.completions.create(
            model="Meta-Llama-3.1-405B-Instruct",
            messages=messages,
            # Retrieve tools installed from Toolhouse
            tools=th.get_tools()
        )
print(response.choices[0].message.content)
