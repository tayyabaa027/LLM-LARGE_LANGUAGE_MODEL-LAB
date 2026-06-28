"""
Phase 6: RAG with Gemini -- a full mini pipeline, no vector database required to learn it.

Install needed packages first with: pip install google-genai python-dotenv numpy

NOTE: the old "google-generativeai" package is now retired by Google.
This script uses the new official replacement package, "google-genai".
"""

import os
# os is used later to read the GEMINI_API_KEY value from the environment

import numpy as np
# numpy is used for the vector math behind cosine similarity

from dotenv import load_dotenv
# dotenv lets this script read variables stored in a local .env file

from google import genai
# this is the new official Gemini SDK, used for both embeddings and chat generation

from google.genai import types
# types holds the config objects we pass alongside our requests

load_dotenv()
# this line actually reads the .env file in the project folder and loads
# its key value pairs into the environment so os.environ can see them

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# this creates the client object used for every request, authenticated with our key from .env

CHAT_MODEL = "gemini-2.5-flash"
# this stores the chat model name used for the final answer generation

EMBEDDING_MODEL = "gemini-embedding-001"
# this stores the Gemini embedding model name used to turn text into vectors
# note: the older "text-embedding-004" name was retired, this is the current replacement


# ---- Your private knowledge the model was never trained on ----
document = """
FAST-NUCES offers a BS Computer Science program across five campuses in Pakistan.
The Islamabad campus has the largest CS faculty and a dedicated AI research lab.
Admission requires passing the NAT or SAT test plus an in-house entrance exam.
Tuition is charged per credit hour and is typically paid in three installments per semester.
The university also offers a need-based financial aid program for underprivileged students.
"""
# this triple quoted string holds the sample private document the pipeline will search over


# ---- Step 1: CHUNKING -- split into small, overlapping, focused pieces ----
def chunk_text(text, chunk_size=25, overlap=5):
    # this breaks the whole document into a simple list of individual words
    words = text.split()

    # this will collect the final chunks, start tracks our current position in the word list
    chunks, start = [], 0

    # this loop keeps slicing chunks until we have walked past the end of the word list
    while start < len(words):
        # this marks where the current chunk should stop
        end = start + chunk_size

        # this joins the sliced words back into a single chunk string and saves it
        chunks.append(" ".join(words[start:end]))

        # this moves the start point forward but steps back by overlap so chunks share some words
        start = end - overlap  # overlap stops ideas from being cut mid-sentence

    # this returns the full list of overlapping text chunks
    return chunks


# this actually runs the chunking function on our sample document
chunks = chunk_text(document)

# this prints how many chunks were produced so we can sanity check the split
print(f"Document split into {len(chunks)} chunks.\n")


# ---- Step 2: EMBEDDINGS -- turn text into vectors of meaning ----
def embed(texts):
    # this will collect one embedding vector per input text
    vectors = []

    # this loop sends each piece of text to the embedding model one at a time
    # doing it one by one avoids batch-size limits that differ between embedding models
    for text in texts:
        # this calls the embedding model and asks for a vector representation of the text
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)

        # this pulls the actual list of numbers out of the first embedding object returned
        vectors.append(response.embeddings[0].values)

    # this converts the list of vectors into a numpy array for easy math later
    return np.array(vectors)


# this embeds every chunk once up front, done once -- reuse these, don't re-embed every query
chunk_vectors = embed(chunks)


# ---- Step 3: VECTOR SEARCH -- find the closest chunks to a query ----
def cosine_similarity(a, b):
    # this measures how similar two vectors are by their angle, not their raw size
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, top_k=2):
    # this turns the user's question into a vector using the same embedding model
    query_vector = embed([query])[0]

    # this computes a similarity score between the query and every chunk vector
    scores = [cosine_similarity(query_vector, v) for v in chunk_vectors]

    # this sorts the scores highest first and keeps only the top_k chunk indices
    ranked = np.argsort(scores)[::-1][:top_k]  # highest similarity first

    # this returns the actual chunk text for each of the top ranked indices
    return [chunks[i] for i in ranked]


# ---- Step 4: AUGMENTATION -- feed retrieved chunks into the final prompt ----
def rag_answer(question):
    # this joins the retrieved chunks into one context block separated by dashes
    context = "\n---\n".join(retrieve(question))

    # this prints the retrieved context so we can see what was actually given to the model
    print("Retrieved context:\n", context, "\n")

    # this builds the config carrying our system instruction for this single call
    config = types.GenerateContentConfig(
        # this forces the model to stick to the given context instead of guessing
        system_instruction="Answer ONLY using the provided context. If it isn't there, say so."
    )

    # this sends the context plus the question together and gets the grounded answer back
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=config,
    )

    # this returns just the plain text of the model's answer
    return response.text


if __name__ == "__main__":
    # this asks a question that should be answerable from the retrieved chunks
    print(rag_answer("Who won the football match yesterday?"))