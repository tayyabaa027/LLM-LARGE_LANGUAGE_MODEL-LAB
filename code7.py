"""
Phase 7: Wrapping Your Gemini LLM Logic in a Production-Ready API

Run locally with:  python -m uvicorn main:app --reload
Install:  pip install fastapi uvicorn google-genai python-dotenv

NOTE: the old "google-generativeai" package is now retired by Google.
This script uses the new official replacement package, "google-genai".
"""

import os
# os is used later to read the GEMINI_API_KEY value from the environment

import time
# time is used to measure how long each request takes, for logging

import logging
# logging is used to record structured information about every request

from dotenv import load_dotenv
# dotenv lets this script read variables stored in a local .env file

from fastapi import FastAPI
# FastAPI is the web framework used to expose our chat logic as HTTP endpoints

from fastapi.responses import StreamingResponse
# StreamingResponse lets us send the reply back to the client piece by piece

from pydantic import BaseModel
# BaseModel is used to define and validate the shape of incoming request bodies

from google import genai
# this is the new official Gemini SDK, used to actually generate the chat replies

load_dotenv()
# this line actually reads the .env file in the project folder and loads
# its key value pairs into the environment so os.environ can see them

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
# this sets up the basic logging format with a timestamp on every log line

logger = logging.getLogger("llm-api")
# this creates a named logger object we will use throughout the file

app = FastAPI(title="LLM CHITCHAT API")
# this creates the actual FastAPI application object that uvicorn will run

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# this creates the client object used for every request, authenticated with our key from .env

MODEL_NAME = "gemini-2.5-flash"
# this stores the model name in one place so it is easy to change later


class ChatRequest(BaseModel):
    # this field holds the actual message text the user wants answered
    message: str
    # this field identifies which user sent the request, defaulting to anonymous
    user_id: str = "anonymous"


@app.post("/chat")
def chat(req: ChatRequest):
    """Non-streaming endpoint: wait for the full answer, then return it once."""
    # this records the start time so we can measure total latency
    start = time.time()
    try:
        # this sends the user's message to Gemini and waits for the complete reply
        response = client.models.generate_content(model=MODEL_NAME, contents=req.message)

        # this pulls just the plain text answer out of the response object
        answer = response.text

        # this tries to read how many tokens were used, since Gemini reports usage metadata
        usage = response.usage_metadata
        total_tokens = usage.total_token_count if usage else "unknown"

        # this writes a structured log line recording the user, latency, tokens, and success
        logger.info(
            f"user={req.user_id} latency={time.time()-start:.2f}s "
            f"tokens={total_tokens} status=success"
        )

        # this returns the final answer back to whoever called this endpoint
        return {"answer": answer}
    except Exception as e:
        # this logs the real error internally if anything above went wrong
        logger.error(f"user={req.user_id} latency={time.time()-start:.2f}s status=error msg={e}")
        # this returns a safe generic message to the caller instead of leaking internal details
        return {"error": "Something went wrong, please try again."}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Streaming endpoint: send tokens to the client as they're generated."""
    def token_generator():
        # this records the start time so we can measure total latency for the stream
        start = time.time()
        try:
            # this calls Gemini's streaming method so chunks arrive as they are produced
            stream = client.models.generate_content_stream(model=MODEL_NAME, contents=req.message)

            # this loop walks through each incoming chunk as Gemini generates it
            for chunk in stream:
                # this checks the chunk actually has text before sending it onward
                if chunk.text:
                    # this sends just that piece of text to the client immediately
                    yield chunk.text

            # this logs a successful completion once the whole stream has finished
            logger.info(f"user={req.user_id} latency={time.time()-start:.2f}s status=streamed")
        except Exception as e:
            # this logs the real error internally if the stream broke partway through
            logger.error(f"user={req.user_id} latency={time.time()-start:.2f}s status=error msg={e}")
            # this sends a clear fallback message to the client instead of dying silently
            yield "\n[error: something went wrong while streaming the response]"

    # this wraps the generator above in a StreamingResponse so FastAPI streams it correctly
    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/health")
def health():
    """Hosting platforms ping this to confirm your app is alive and responsive."""
    # this simply returns ok, proving the process is alive and responding to requests
    return {"status": "ok"}

# To deploy this to Railway: push the project (including a requirements.txt listing
# fastapi, uvicorn, google-genai, and python-dotenv) to a GitHub repo, create a new
# Railway project from that repo, set GEMINI_API_KEY as an environment variable in
# Railway's dashboard (never in your code), and let it build and deploy automatically.