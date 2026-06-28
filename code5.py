"""
Phase 5: Tool Calling with Gemini -- Let the Model Trigger Real Functions

The model NEVER executes code. It only names a function and suggests arguments.
Your code decides whether those arguments are safe to actually run.

Install needed packages first with: pip install google-genai python-dotenv

NOTE: the old "google-generativeai" package is now retired by Google.
This script uses the new official replacement package, "google-genai",
imported as "from google import genai".
"""

import os
# os is used later to read the GEMINI_API_KEY value from the environment

from dotenv import load_dotenv
# dotenv lets this script read variables stored in a local .env file

from google import genai
# this is the new official Gemini SDK, used to talk to the Gemini API

from google.genai import types
# types holds the building blocks we need to describe tools and pass config options

load_dotenv()
# this line actually reads the .env file in the project folder and loads
# its key value pairs into the environment so os.environ can see them

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# this creates the client object used for every request, authenticated with our key from .env

MODEL = "gemini-2.5-flash"
# this stores the model name in one place so it is easy to change later


# ---- Step 1: a REAL Python function the model is allowed to trigger ----
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """A toy converter -- swap this for a real exchange-rate API in production."""
    # this dictionary stores fake fixed conversion rates for the demo
    fake_rates = {"USD_PKR": 280.0, "PKR_USD": 1 / 280.0}

    # this builds the lookup key like USD_PKR from the two currency codes
    key = f"{from_currency.upper()}_{to_currency.upper()}"

    # this checks if we actually have a rate for that currency pair
    if key not in fake_rates:
        # if the pair is missing, return an error dictionary instead of crashing
        return {"error": f"No rate available for {key}"}

    # this multiplies the amount by the rate and rounds it to 2 decimal places
    return {"result": round(amount * fake_rates[key], 2), "pair": key}


# ---- Step 2: describe that function to the model as a FunctionDeclaration ----
# the new SDK wants a plain dictionary style schema, using lowercase JSON Schema types
convert_currency_declaration = types.FunctionDeclaration(
    name="convert_currency",
    # this is the plain English description the model uses to decide when to call this tool
    description="Convert an amount of money from one currency to another.",
    parameters={
        "type": "object",
        # this lists every parameter the model must fill in to call the function
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert"},
            "from_currency": {"type": "string", "description": "3-letter currency code, e.g. USD"},
            "to_currency": {"type": "string", "description": "3-letter currency code, e.g. PKR"},
        },
        # this tells the model all three fields are mandatory, none can be skipped
        "required": ["amount", "from_currency", "to_currency"],
    },
)

# this wraps the declaration into a Tool object the model is allowed to use
tool = types.Tool(function_declarations=[convert_currency_declaration])

# this bundles the tool into the generation config we pass on every request
config = types.GenerateContentConfig(tools=[tool])

# this dictionary maps the function name string back to the real Python function
AVAILABLE_FUNCTIONS = {"convert_currency": convert_currency}


def run_tool_loop(user_question):
    # this starts the conversation history as a plain list of message turns
    contents = [
        types.Content(role="user", parts=[types.Part(text=user_question)])
    ]

    # this sends the conversation so far to Gemini, along with the tool config
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    # this grabs the first candidate part of the reply to inspect what the model wants to do
    part = response.candidates[0].content.parts[0]

    # this checks if the model's reply is a function call request rather than plain text
    if part.function_call is None:
        # if there is no function call, the model answered directly, so just return that text
        return response.text

    # this reads the name of the function the model wants to call
    fn_name = part.function_call.name

    # this converts the model's suggested arguments into a normal Python dictionary
    raw_args = dict(part.function_call.args)

    # --- SAFETY CHECK before executing anything the model suggested ---
    if fn_name not in AVAILABLE_FUNCTIONS:
        # this blocks execution if the model invented a function name we do not have
        result = {"error": "Unknown function -- refusing to execute."}
    elif not isinstance(raw_args.get("amount"), (int, float)) or raw_args["amount"] < 0:
        # this blocks execution if the amount is missing, the wrong type, or negative
        result = {"error": "Invalid amount -- refusing to execute."}
    else:
        # this only runs the real function once both safety checks above have passed
        result = AVAILABLE_FUNCTIONS[fn_name](**raw_args)

    # this appends the model's own function call turn to the conversation history
    contents.append(response.candidates[0].content)

    # this appends our function result as a new turn, tagged with the matching function name
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=fn_name,
                    # this packages the dictionary result so Gemini can read it as structured data
                    response={"result": result},
                )
            ],
        )
    )

    # this calls the model again, now with the real tool result available in the history
    final = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    # this returns the model's final natural language answer, now grounded in the real result
    return final.text


if __name__ == "__main__":
    # this runs a question that should trigger the currency conversion tool
    print(run_tool_loop("If a freelancer earns 500 USD, how many PKR is that?"))
    # this runs a question that needs no tool at all, just general knowledge
    print(run_tool_loop("What's the capital of Pakistan?"))  # no tool needed here