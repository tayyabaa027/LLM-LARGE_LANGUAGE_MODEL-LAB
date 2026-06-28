"""
Phase 8: Hallucination Guardrails and Pakistan-Specific Concerns, using Gemini

Install needed packages first with: pip install google generativeai python dotenv
"""

import os
# os is used later to read the GEMINI_API_KEY value from the environment

import re
# re is used to write the regular expression patterns that find PII in text

from dotenv import load_dotenv
# dotenv lets this script read variables stored in a local .env file

import google.generativeai as genai
# this is the official Gemini SDK, used for both chat generation and token counting

load_dotenv()
# this line actually reads the .env file in the project folder and loads
# its key value pairs into the environment so os.environ can see them

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# this configures the Gemini client using the key that was just loaded from .env

MODEL_NAME = "gemini-2.5-flash"
# this stores the model name in one place so it is easy to change later

model = genai.GenerativeModel(model_name=MODEL_NAME)
# this creates the reusable Gemini model object used for both generation and token counting


# ---- 1. HALLUCINATION GUARDRAIL: make the model flag its own uncertainty ----
def ask_with_confidence(question):
    # this builds a model instance with a system instruction asking it to self rate confidence
    guarded_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=(
            "Answer the question. Then on a new line write "
            "CONFIDENCE: high / medium / low, based on how certain you are "
            "this is factually correct rather than a plausible-sounding guess."
        ),
    )

    # this sends the actual question to the guarded model and gets the reply back
    response = guarded_model.generate_content(question)

    # this returns just the plain text of the answer plus its confidence line
    return response.text


# ---- 2. PII REDACTION: never send raw Pakistani identifiers to a third-party API ----
CNIC_PATTERN = re.compile(r"\b\d{5}-\d{7}-\d\b")    # e.g. 35202-1234567-1
# this pattern matches the standard 13 digit CNIC format with dashes in the right places

PHONE_PATTERN = re.compile(r"\b03\d{2}-?\d{7}\b")    # e.g. 0312-3456789
# this pattern matches an 11 digit Pakistani mobile number, with or without a dash

def redact_pii(text):
    # this replaces any CNIC number found in the text with a safe placeholder
    text = CNIC_PATTERN.sub("[CNIC-REDACTED]", text)
    # this replaces any phone number found in the text with a safe placeholder
    text = PHONE_PATTERN.sub("[PHONE-REDACTED]", text)
    # this returns the cleaned text, now safe to send onward to an external API
    return text


# this is a sample message containing fake but realistic looking PII for testing
raw_message = "My CNIC is 35202-1234567-1 and my number is 0312-3456789, please help me draft a complaint."

# this prints the original message so we can compare it against the redacted version
print("Before:", raw_message)
# this prints the message after both patterns have been redacted out of it
print("After: ", redact_pii(raw_message))


# ---- 3. TOKEN COST: same meaning, different script, different cost ----
samples = {
    "English":    "Please tell me the bus fare from Lahore to Islamabad.",
    "Roman Urdu": "Lahore se Islamabad tak bus ka kiraya kya hai?",
    "Urdu script": "لاہور سے اسلام آباد تک بس کا کرایہ کیا ہے؟",
}
# this dictionary holds the same question written in three different scripts for comparison

print("\nToken cost comparison for the SAME question, three ways:")
# this loop checks how many tokens Gemini actually uses for each version of the question
for label, text in samples.items():
    # this calls Gemini's own count_tokens method instead of a separate offline tokenizer
    result = model.count_tokens(text)
    # this pulls the total token count out of the result object
    n = result.total_tokens
    # this prints the script label, its token count, and the original text for comparison
    print(f"  {label:<12} -> {n} tokens   ({text})")


if __name__ == "__main__":
    # this runs the confidence guardrail on a sample question to see it in action
    print("\n", ask_with_confidence("What year was FAST-NUCES founded?"))