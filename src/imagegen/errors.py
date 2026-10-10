"""Typed failures raised by the imagegen service.

Every failure that a client can trigger is represented by an :class:`ApiError`
subclass so the HTTP layer can render a single, OpenAI-shaped error body from
one place instead of sprinkling status codes across the routes.
"""

from __future__ import annotations


class ApiError(Exception):
    """Base class for errors that are reported to the HTTP client.

    Attributes:
        message: Human readable description, placed in ``error.message``.
        status_code: HTTP status returned for this error.
        error_type: OpenAI error discriminator, e.g. ``invalid_request_error``.
        code: Stable machine readable identifier clients may branch on.
        param: Name of the offending request field, when there is one.
    """

    message: str = "internal error"
    status_code: int = 500
    error_type: str = "api_error"
    code: str | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message if message is not None else self.message
        self.param = param
        self.code = code if code is not None else self.code
        super().__init__(self.message)

    def to_openai_body(self) -> dict[str, object]:
        """Render the error using the OpenAI error envelope."""
        error: dict[str, object] = {"message": self.message, "type": self.error_type}
        if self.param is not None:
            error["param"] = self.param
        if self.code is not None:
            error["code"] = self.code
        return {"error": error}


class InvalidRequestError(ApiError):
    """The request is syntactically fine but semantically unacceptable."""

    status_code = 400
    error_type = "invalid_request_error"


class AuthenticationError(ApiError):
    """The bearer token is missing or unknown."""

    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"


class ModelNotFoundError(ApiError):
    """The requested ``model`` is not registered in the service config."""

    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class BackendUnavailableError(ApiError):
    """No inference backend could be made available for the model."""

    status_code = 503
    error_type = "api_error"
    code = "backend_unavailable"


class BackendReturnedError(ApiError):
    """The backend answered with a non-success status for a valid request."""

    status_code = 502
    error_type = "api_error"
    code = "backend_error"


class QueueFullError(ApiError):
    """Too many requests are already waiting for the single GPU resource."""

    status_code = 429
    error_type = "rate_limit_error"
    code = "queue_full"


class ServiceConfigurationError(Exception):
    """The service configuration is invalid; raised during startup only.

    This is deliberately not an :class:`ApiError`: a broken config must stop the
    process rather than surface as a per-request failure.
    """
