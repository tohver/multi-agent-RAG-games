"""Workaround for an unreliable OpenAI proxy.

The proxy this project was developed against intermittently answers with an
Apache style HTML "404 Not Found" page instead of a JSON response - measured at
roughly 40% of calls. It affects `/chat/completions` and `/embeddings` alike, so
the retry is installed once at the HTTP layer rather than in every caller.

Only HTML 404s are retried. A genuine JSON 404 from the API is a real error and
is passed straight through.
"""

import time

import httpx

_MARKER = "_retries_html_404"


def install_html_404_retry(max_attempts: int = 5, base_delay: float = 0.4) -> bool:
    """Patch `httpx.Client.send` to retry HTML 404 responses.

    Idempotent: calling it twice leaves a single layer of retries in place.

    Args:
        max_attempts: How many times a request may be sent in total.
        base_delay: First backoff delay in seconds; doubles on each retry.

    Returns:
        True if the patch was installed, False if it was already present.
    """
    if getattr(httpx.Client.send, _MARKER, False):
        return False

    original_send = httpx.Client.send

    def send_retrying_html_404(self, request, **kwargs):
        """Send the request, retrying while the response is an HTML 404."""
        for attempt in range(max_attempts):
            response = original_send(self, request, **kwargs)

            # Streaming responses must not be read here, and a non-404 is final.
            if response.status_code != 404 or kwargs.get("stream"):
                return response

            response.read()
            if b"<html" not in response.content.lower():
                return response  # a real API 404 - do not mask it

            response.close()
            time.sleep(base_delay * 2**attempt)

        return response

    setattr(send_retrying_html_404, _MARKER, True)
    httpx.Client.send = send_retrying_html_404
    return True
