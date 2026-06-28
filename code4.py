"""
Phase 4: Memory Is an Illusion. Managing Conversation History
This version uses Gemini instead of OpenAI.
"""

import os
# Imports the os module so the script can read values from the operating system, such as environment variables.

import tiktoken
# Imports tiktoken, a tokenizer library originally built for OpenAI models. It is used here only to get a rough, approximate token count, since Gemini does not ship its own offline tokenizer library. The count will not be exact for Gemini, but it is close enough to demonstrate trimming.

from dotenv import load_dotenv
# Imports the load_dotenv function from the python dotenv package so this file can load the .env file on its own.

from google import genai
# Imports the genai module from the google package so the script can talk to Google's Gemini servers.

from google.genai import types
# Imports the types module from genai. This gives access to GenerateContentConfig and Content, needed for system instructions and multi turn conversations.

load_dotenv()
# Runs the function that reads the .env file in this folder and copies its key and value pairs into the environment for this program to use.

encoder = tiktoken.get_encoding("cl100k_base")
# Loads the cl100k_base encoding rules into a reusable encoder object. This is the encoding OpenAI models use, borrowed here purely as an approximation since Gemini uses a different tokenizer internally.

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
# Creates a client object connected to Gemini, authenticating with the key pulled from the environment variable that load_dotenv loaded above.

MODEL = "gemini-3.1-flash-lite"
# Stores the model name in one place so it only needs to be changed in one spot if you want to try a different Gemini model.


def count_tokens(messages):
    """Rough token count for a list of role and content dicts."""
    # This docstring explains what the function does for anyone reading the code later.

    total = 0
    # Starts a running total at zero before adding up every message.

    for m in messages:
        # Loops through each message dictionary in the list that was passed in.

        total += len(encoder.encode(m["content"])) + 4
        # Encodes the text into tokens, counts how many there are, then adds 4 as a rough estimate for the small overhead every message carries beyond just its raw text.

    return total
    # Sends the final total back to whatever code called this function.


class ConversationManager:
    """Keeps chat history under a token budget so you never blow the context window."""
    # This docstring explains the overall purpose of the class for anyone reading the code later.

    def __init__(self, system_prompt, max_tokens=300):
        # Defines the setup method that runs automatically whenever a new ConversationManager is created.

        self.max_tokens = max_tokens
        # Stores the token budget on the object itself so other methods in this class can check against it later.

        self.system_prompt = system_prompt
        # Stores the system prompt as a plain string on the object. Gemini takes the system instruction as its own separate setting, not as an entry inside the conversation list, so it is kept apart from history here.

        self.history = []
        # Starts an empty list that will hold every user and model turn, but never the system prompt itself.

    def add_user_message(self, content):
        # Defines a method for adding a new message from the human user into the conversation.

        self.history.append({"role": "user", "content": content})
        # Adds a dictionary representing this turn to the end of the history list, tagged with role user.

        self._trim()
        # Immediately checks whether the conversation has grown past the token budget and trims it if needed.

    def add_assistant_message(self, content):
        # Defines a method for adding a new reply from the model into the conversation.

        self.history.append({"role": "model", "content": content})
        # Adds a dictionary representing this turn to the history list. Gemini calls the assistant's own role model, not assistant like OpenAI does, so that exact word is used here.

        self._trim()
        # Checks the token budget again after adding this new message.

    def _trim(self):
        """Sliding window trim, drop the oldest turns first until we fit the budget."""
        # This docstring explains the trimming strategy used by this method.

        while count_tokens([{"role": "system", "content": self.system_prompt}] + self.history) > self.max_tokens and len(self.history) > 2:
            # Keeps looping as long as two things are both true: the conversation, including the system prompt, is over the token budget, and there are still more than 2 messages left so we do not trim away everything.

            removed = self.history.pop(0)
            # Removes and returns the very oldest message in the history list, which is the first item, since index 0 is the front of the list.

            print(f"  [trimmed] dropped {removed['role']} message: {removed['content'][:40]!r}...")
            # Prints a short notice showing which message got dropped, including its role and the first 40 characters of its text, so you can watch trimming happen as you run the script.

    def get_contents(self):
        # Defines a method that converts the internal history list into the exact format Gemini expects for its contents argument.

        contents = []
        # Starts an empty list that will be filled with properly formatted turns.

        for m in self.history:
            # Loops through every stored turn in the history list.

            contents.append(types.Content(role=m["role"], parts=[types.Part(text=m["content"])]))
            # Wraps each turn's role and text into the Content and Part objects Gemini's SDK requires, then adds it to the list being built.

        return contents
        # Sends the finished list of Content objects back to whatever code called this method.

    def ask(self, user_text):
        # Defines the main method you actually call from outside the class to have a turn of conversation.

        self.add_user_message(user_text)
        # Adds the new question into history and trims if needed, before sending anything to Gemini.

        response = client.models.generate_content(
            # Sends the request to Gemini and waits for a reply.

            model=MODEL,
            # Tells Gemini which model should answer, using the constant defined above.

            contents=self.get_contents(),
            # Passes in the full trimmed conversation history, properly formatted, so Gemini sees the whole context, not just the latest question.

            config=types.GenerateContentConfig(system_instruction=self.system_prompt),
            # Passes the system prompt in separately as Gemini expects, rather than mixing it into the contents list.
        )
        reply = response.text
        # Pulls just the text of Gemini's answer out of the larger response object.

        self.add_assistant_message(reply)
        # Stores Gemini's reply into history as a model turn, so it becomes part of the context for the next question.

        return reply
        # Sends the reply text back to whatever code called ask(), so it can be printed or used elsewhere.


if __name__ == "__main__":
    # This line makes sure the code below only runs when this file is executed directly, not when it gets imported into another file.

    convo = ConversationManager(
        # Creates a new ConversationManager object and stores it in the variable convo.

        system_prompt="You are a friendly assistant who remembers context within this chat only.",
        # Sets the personality and behavior instruction that will be sent to Gemini on every single call.

        max_tokens=300,
        # Sets a deliberately small token budget so trimming actually happens during this short demo and you can see it occur.
    )

    questions = [
        # Builds a list of three questions that will be asked one after another in the same conversation.

        "My name is Hira and I'm building a chatbot for my final-year project.",
        # The first message, which introduces a fact, the user's name, that later messages will test whether the bot still remembers.

        "What's a good database for storing chat history?",
        # The second message, an unrelated question that adds more bulk to the conversation, pushing it closer to the token budget.

        "Can you remind me what my name is?",
        # The third message, which tests whether the name from the first message has already been trimmed out of history by this point.
    ]

    for q in questions:
        # Loops through each question in the list, one at a time, in order.

        print(f"\nUSER: {q}")
        # Prints the question being asked, labeled clearly so the terminal output is easy to follow.

        print("BOT: ", convo.ask(q))
        # Calls ask() with this question, which sends it to Gemini, stores the new reply, and prints that reply labeled as coming from the bot.