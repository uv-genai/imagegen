"""In-memory store backing ``response_format=url`` image responses.

OpenAI returns time-limited URLs. Reproducing that here needs somewhere to keep
the bytes between the generation and the client's download, so images live in a
bounded, expiring cache rather than on disk: nothing outlives the process, which
is the right default for picture data a client is expected to fetch immediately.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredImage:
    """A generated image held for URL retrieval."""

    blob_id: str
    content: bytes
    media_type: str
    expires_at: float

    def is_expired(self, now: float) -> bool:
        """True when this entry is past its expiry instant."""
        return now >= self.expires_at


#: Mapping from the API's ``output_format`` values to media types.
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


class ImageBlobStore:
    """Bounded LRU-with-TTL store for images addressed by an opaque id."""

    def __init__(self, ttl_seconds: int, capacity: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._capacity = capacity
        self._entries: dict[str, StoredImage] = {}

    def __len__(self) -> int:
        """Number of live entries currently stored."""
        return len(self._entries)

    def store(self, content: bytes, output_format: str) -> StoredImage:
        """Keep ``content`` for ``ttl_seconds`` and return its handle.

        Insertion evicts expired entries first, then the oldest ones, so the
        store can never grow past ``capacity`` even under sustained load.
        """
        self._evict(time.monotonic())
        entry = StoredImage(
            blob_id=secrets.token_urlsafe(16),
            content=content,
            media_type=MEDIA_TYPES[output_format],
            expires_at=time.monotonic() + self._ttl_seconds,
        )
        self._entries[entry.blob_id] = entry
        return entry

    def get(self, blob_id: str) -> StoredImage | None:
        """Return a live entry by id, or None when absent or expired."""
        entry = self._entries.get(blob_id)
        if entry is None:
            return None
        if entry.is_expired(time.monotonic()):
            del self._entries[blob_id]
            return None
        return entry

    def _evict(self, now: float) -> None:
        """Drop expired entries, then oldest entries until under capacity."""
        for blob_id in [key for key, value in self._entries.items() if value.is_expired(now)]:
            del self._entries[blob_id]
        while len(self._entries) >= self._capacity:
            oldest = next(iter(self._entries))
            del self._entries[oldest]
