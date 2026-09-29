# Sends one question to a local Ollama model and returns the reply text.

import requests

from config import OLLAMA_URL, TIMEOUT


def chat(model, system, user, temperature, seed, max_new_tokens):
    messages = []
    if system:  # the "none" condition has no system prompt
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    request = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": max_new_tokens},
    }
    response = requests.post(OLLAMA_URL, json=request, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]["content"]
