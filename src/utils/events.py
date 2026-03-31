"""Redis Streams event bus for inter-component communication."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import redis.asyncio as redis

from src.utils.logging import get_logger

log = get_logger(__name__)


class EventBus:
    """Pub/sub event bus backed by Redis Streams."""

    def __init__(self, redis_url: str):
        self._redis: redis.Redis | None = None
        self._redis_url = redis_url

    async def connect(self) -> None:
        self._redis = redis.from_url(self._redis_url, decode_responses=True)
        await self._redis.ping()
        log.info("event_bus_connected")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def publish(self, stream: str, data: dict[str, Any]) -> str:
        """Publish an event to a stream. Returns the message ID."""
        # Serialize nested objects to JSON strings
        flat = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v)

        msg_id = await self._redis.xadd(stream, flat, maxlen=10000)
        return msg_id

    async def subscribe(
        self, stream: str, last_id: str = "$", block_ms: int = 1000
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Subscribe to a stream, yielding (message_id, data) tuples."""
        current_id = last_id
        while True:
            try:
                results = await self._redis.xread(
                    {stream: current_id}, block=block_ms, count=100
                )
                if results:
                    for _stream_name, messages in results:
                        for msg_id, data in messages:
                            current_id = msg_id
                            # Deserialize JSON strings back
                            parsed = {}
                            for k, v in data.items():
                                try:
                                    parsed[k] = json.loads(v)
                                except (json.JSONDecodeError, TypeError):
                                    parsed[k] = v
                            yield msg_id, parsed
            except redis.ConnectionError:
                log.warning("event_bus_reconnecting")
                await self.connect()

    async def get_latest(self, stream: str, count: int = 1) -> list[dict]:
        """Get the latest N messages from a stream."""
        results = await self._redis.xrevrange(stream, count=count)
        messages = []
        for _msg_id, data in results:
            parsed = {}
            for k, v in data.items():
                try:
                    parsed[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed[k] = v
            messages.append(parsed)
        return messages
