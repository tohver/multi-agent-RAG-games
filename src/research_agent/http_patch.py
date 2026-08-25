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
    '''
    In plain English: this makes the program survive a flaky server.

    Some proxies occasionally hand back a web page saying "404 Not Found"
    instead of the answer you asked for. Nothing is actually wrong - ask again
    and it works. Rather than wrapping every single API call in its own retry,
    this function reaches into the HTTP library once and teaches it to retry
    that specific failure everywhere, for every call the program will ever make.

    Output: `True` if the patch was installed, `False` if it was already there.
    That return value mostly protects against installing the retry twice, which
    would stack retries on top of retries. Called once from
    `build_application`, before anything else touches the network.
    '''
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
        '''
        In plain English: the replacement for the HTTP library's "send" step.

        It sends the request as normal. If the reply is an HTML page with a 404,
        it waits a moment and tries again - doubling the wait each time, so a
        server that is genuinely struggling is not hammered. Any other reply,
        including a real JSON 404 from the API, is handed back untouched, so
        genuine errors stay visible instead of being silently retried away.

        Output: the HTTP response, exactly as the caller expects. Nobody calls
        this directly; it takes the place of the library's own method, so every
        request in the program passes through it without knowing.
        '''
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
