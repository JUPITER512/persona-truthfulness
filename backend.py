import requests
from config import OLLAMA_URL, TIMEOUT


def _ollama(model, messages, temperature, seed, max_new_tokens):
    r = requests.post(OLLAMA_URL,
                      json={
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed,
        "num_predict": max_new_tokens},
        },
    timeout=TIMEOUT)
    r.raise_for_status()

    return r.json()["message"]["content"]


def chat(model, system, user, temperature, seed, max_new_tokens):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user})
    return _ollama(model, messages, temperature, seed, max_new_tokens)
