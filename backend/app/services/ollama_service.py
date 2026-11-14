"""OllamaService: all Ollama communication lives here.

API routes never talk to Ollama directly — they use this service instead.
Every method fails gracefully (no exceptions bubble up to callers) so the
backend stays up even when Ollama is unreachable.
"""

import httpx

from app.config import settings


class OllamaService:
    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _installed_model_names(self, client: httpx.Client) -> list[str]:
        response = client.get("/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [entry.get("name", "") for entry in models]

    @staticmethod
    def _model_matches(requested: str, installed: str) -> bool:
        """Match 'llama3.2:latest' against 'llama3.2' and vice versa."""
        if requested == installed:
            return True
        requested_name, _, requested_tag = requested.partition(":")
        installed_name, _, installed_tag = installed.partition(":")
        if requested_name != installed_name:
            return False
        # Treat an untagged request ('llama3.2') as '<name>:latest'.
        return (requested_tag or "latest") == (installed_tag or "latest")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Generate a response from the chat model via Ollama's /api/chat.

        Raises RuntimeError with a clear message on any failure; callers
        decide how to surface it (no stack traces leak to users).
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        effective_timeout = timeout if timeout is not None else settings.ollama_chat_timeout
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            with httpx.Client(base_url=self.base_url, timeout=effective_timeout) as client:
                response = client.post(
                    "/api/chat",
                    json={"model": self.model, "messages": messages, "stream": False},
                )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama did not respond within {effective_timeout}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not reach Ollama: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama /api/chat responded with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError) as exc:
            raise RuntimeError("Ollama returned an unexpected response format.") from exc

        content = content.strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    def check_connection(self) -> bool:
        """True if the Ollama server responds on this machine."""
        try:
            with self._client() as client:
                response = client.get("/api/version")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def is_model_available(self) -> bool:
        """True if the configured generation model is installed in Ollama."""
        try:
            with self._client() as client:
                names = self._installed_model_names(client)
            return any(self._model_matches(self.model, name) for name in names)
        except (httpx.HTTPError, ValueError):
            # ValueError covers invalid JSON in the response body.
            return False

    def status(self) -> dict:
        """Structured status for the /api/health endpoint. Never raises."""
        result = {
            "reachable": False,
            "model": self.model,
            "model_available": False,
            "error": None,
        }
        try:
            with self._client() as client:
                version_response = client.get("/api/version")
                if version_response.status_code != 200:
                    result["error"] = (
                        f"Ollama responded with HTTP {version_response.status_code}."
                    )
                    return result

                result["reachable"] = True

                tags_response = client.get("/api/tags")
                if tags_response.status_code != 200:
                    result["error"] = (
                        f"Ollama /api/tags responded with HTTP {tags_response.status_code}."
                    )
                    return result

                names = [
                    entry.get("name", "")
                    for entry in tags_response.json().get("models", [])
                ]
                result["model_available"] = any(
                    self._model_matches(self.model, name) for name in names
                )
                if not result["model_available"]:
                    result["error"] = (
                        f"Model '{self.model}' is not installed. "
                        f"Run: ollama pull {self.model}"
                    )
                return result
        except httpx.ConnectError:
            result["error"] = (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )
        except httpx.TimeoutException:
            result["error"] = (
                f"Ollama at {self.base_url} did not respond within {self.timeout}s."
            )
        except httpx.HTTPError:
            result["error"] = f"Could not reach Ollama at {self.base_url}."
        except ValueError:
            result["error"] = "Ollama returned an unexpected (non-JSON) response."
        return result


# Shared instance built from central configuration.
ollama_service = OllamaService(
    base_url=settings.ollama_base_url,
    model=settings.ollama_chat_model,
    timeout=settings.ollama_timeout,
)
