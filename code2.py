"""
Phase 2: Your First Calls to Three Different LLM Providers
Install:  pip install openai anthropic google-genai
Before running, set these as environment variables in your terminal:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_API_KEY="AIza..."
(On Windows PowerShell, use: $env:OPENAI_API_KEY="sk-...")
This teaches you how to connect your Python script to OpenAI, Claude, and
Gemini using your private API keys to send a prompt and receive a real AI response.
It specifically demonstrates the technical "shape" required by
each provider, such as how to format system instructions and how to extract
metadata like total tokens used for each call.
"""

import os
# Imports the os module so the script can read values from the operating system, such as environment variables.

from dotenv import load_dotenv
# Imports the load_dotenv function from the python dotenv package so the script can load secret values from a .env file.

from openai import OpenAI
# Imports the OpenAI class from the openai package so the script can build a client that talks to OpenAI's servers.

load_dotenv()  # loads environment variables from .env file
# Runs the function that reads the .env file sitting in this folder and copies its key and value pairs into the environment for this program to use.

PROMPT = "In one sentence, why do programmers love Python?"
# Stores the question that will be sent to every provider, kept in one variable so all three calls use the exact same text.


# SECTION 1: OPENAI
def call_openai():
    # Defines a function named call_openai that groups together everything needed to send one request to OpenAI and print the answer.

    from openai import OpenAI
    # Imports OpenAI again here. This line is not actually needed since it was already imported above, but it does not cause an error.

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # Creates a client object connected to OpenAI, authenticating with the key pulled from the environment variable that load_dotenv loaded earlier.

    response = client.chat.completions.create(
        # Sends the actual request to OpenAI and waits for the model to generate a reply. The full reply gets stored in the variable named response.

        model="gpt-5.4-mini",  # cheap, fast, good for learning, check docs for newer options
        # Picks which OpenAI model answers the prompt. This one is a small, inexpensive model that is good for testing.

        messages=[
            # This list holds the conversation. OpenAI expects every entry to have a role and a piece of content.

            {"role": "system", "content": "You are a witty CS professor."},
            # This entry sets the personality and behavior the assistant should follow before it even sees the question.

            {"role": "user", "content": PROMPT},
            # This entry is the actual question being asked, pulled from the PROMPT variable defined earlier.
        ],
        temperature=0.7,
        # Controls how random or creative the answer is. A higher number gives more varied answers, a lower number gives more predictable ones.

        max_tokens=100,
        # Sets the maximum length of the answer in tokens, so the response cannot run on forever and cost more than expected.
    )
    text = response.choices[0].message.content
    # Pulls just the text of the answer out of the larger response object that OpenAI sent back.

    tokens_used = response.usage.total_tokens
    # Pulls the total number of tokens this single request used, which is useful for tracking cost.

    print(f"[OpenAI]  {text}  (tokens used: {tokens_used})")
    # Prints the answer to the terminal along with how many tokens it used, labeled so it is clear the answer came from OpenAI.


# SECTION 2: ANTHROPIC (Claude)
# This whole section is written as plain comments on purpose, so it does not run and does not need an API key yet.
# Remove the pound signs in front of each line below once you are ready to actually call Claude.
#
# def call_claude():
#     # Defines a function that groups together everything needed to send one request to Claude and print the answer.
#
#     import anthropic
#     # Imports the anthropic package so the script can talk to Claude's servers.
#
#     client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
#     # Creates a client object connected to Anthropic, authenticating with the key pulled from the environment.
#
#     response = client.messages.create(
#         # Sends the request to Claude and waits for a reply.
#
#         model="claude-haiku-4-5-20251001",
#         # Picks which Claude model answers the prompt. This is the cheapest current model, good for tutorials.
#
#         system="You are a witty CS professor.",
#         # Claude takes the system instruction as its own separate setting, not as an entry inside the messages list.
#
#         max_tokens=100,
#         # Sets the maximum length of the answer in tokens.
#
#         messages=[
#             {"role": "user", "content": PROMPT},
#             # The actual question being asked.
#         ],
#     )
#     text = response.content[0].text
#     # Pulls the text of the answer out of Claude's response object.
#
#     tokens_used = response.usage.input_tokens + response.usage.output_tokens
#     # Claude reports input tokens and output tokens separately, so they are added together here to get the total.
#
#     print(f"[Claude]  {text}  (tokens used: {tokens_used})")
#     # Prints the answer along with the token cost, labeled so it is clear the answer came from Claude.


# SECTION 3: GOOGLE GEMINI
def call_gemini():
    # Defines a function that groups together everything needed to send one request to Gemini and print the answer.

    from google import genai
    # Imports the genai package so the script can talk to Google's Gemini servers.

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    # Creates a client object connected to Gemini, authenticating with the key pulled from the environment.

    response = client.models.generate_content(
        # Sends the request to Gemini and waits for a reply.

        model="gemini-3.1-flash-lite",
        # Picks which Gemini model answers the prompt. This one is fast and inexpensive.

        contents=f"You are a witty CS professor. {PROMPT}",
        # Gemini does not split a system instruction and a user question the way OpenAI does, so both are combined into a single piece of text here.
    )
    print(f"[Gemini]  {response.text}")
    # Prints Gemini's answer to the terminal, labeled so it is clear the answer came from Gemini.


if __name__ == "__main__":
    # This line makes sure the code below only runs when this file is executed directly, not when it gets imported into another file.

    for name, fn in [("OpenAI", call_openai), ("Gemini", call_gemini)]:
        # Loops through a list of pairs. Each pair holds a label and the function to call for that provider. Claude is left out of this list for now since its function is commented out above.

        try:
            # Starts a block that will catch any error, so one provider failing does not stop the whole script from finishing.

            fn()
            # Actually calls the function for this provider, which sends the request and prints the result.

        except Exception as e:
            # If anything goes wrong inside fn(), this line catches the error instead of letting the program crash.

            print(f"[{name}] Skipped: {e}")
            # Prints which provider failed and why, then the loop moves on to the next provider in the list.