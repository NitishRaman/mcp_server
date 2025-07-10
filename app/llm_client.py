# ✅ llm_client.py — Unified LLMClient with GPT4All, Ollama, and HuggingFace support

from app.core.config import LLM_BACKEND, LLM_MODEL, HF_API_KEY, HF_ENDPOINT
import os
import openai
import requests


# Conditionally import based on backend
if LLM_BACKEND == "ollama":
    from ollama import chat
elif LLM_BACKEND == "gpt4all":
    from gpt4all import GPT4All

# --- Class: LLMClient ---
class LLMClient:
    """
    A unified abstraction for interacting with different LLM backends.

    Supports:
    - GPT4All (local model via gguf)
    - Ollama (local lightweight models)
    - Hugging Face (remote API, free-tier available)
    - OpenAI (via API key)

    Reads model and backend config from app.core.config:
    - LLM_BACKEND: "gpt4all", "ollama", "openai", or "huggingface"
    - LLM_MODEL: path or name of the model
    """

    def __init__(self):
        self.backend = LLM_BACKEND
        self.model_name = LLM_MODEL
        self.api_key = os.getenv("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")  # Securely load OpenAI API key from env var

        if self.backend == "gpt4all":
            self.model = GPT4All(self.model_name)

    # --- Method: ask ---
    def ask(self, prompt: str) -> str:
        """
        Ask the model a question using a single prompt. Delegates to chat_completion.
        """
        if self.backend == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key not set.")

            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    api_key=self.api_key,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.choices[0].message.content
                return content if content else "⚠️ Empty response from LLM."

            except Exception as e:
                return f"❌ OpenAI error: {e}"

        else:
            return self.chat_completion([
                {"role": "user", "content": prompt}
            ])

    # --- Method: chat_completion ---
    def chat_completion(self, messages, max_tokens=256):
        """
        Generate a response from the selected LLM backend.

        Args:
            messages (list): A list of dicts with 'role' and 'content'
            max_tokens (int): Max tokens to return

        Returns:
            str: The response text from the model
        """

        if self.backend == "ollama":
            # Use Ollama client for chat completion
            response = chat(model=self.model_name, messages=messages)
            return response['message']['content']

        elif self.backend == "gpt4all":
            # Turn messages into a single prompt for GPT4All
            prompt = self._messages_to_prompt(messages)
            with self.model.chat_session():
                out = self.model.generate(prompt, max_tokens=max_tokens)
            return out.strip()

        elif self.backend == "huggingface":
            try:
                prompt = self._messages_to_prompt(messages)
                response = requests.post(
                    HF_ENDPOINT,
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
                )
                result = response.json()

                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
                else:
                    return str(result)
            except Exception as e:
                return f"❌ HF API Error: {e}"

        else:
            return f"❌ Unknown backend: {self.backend}"

    # --- Method: _messages_to_prompt ---
    def _messages_to_prompt(self, messages):
        """
        Helper to convert list of chat messages to flat prompt.

        Args:
            messages (list): Chat-style messages with 'role' and 'content'

        Returns:
            str: A prompt string
        """
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
