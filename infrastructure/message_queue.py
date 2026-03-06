import asyncio
import json
from typing import Callable, Optional
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger("Infrastructure.MessageQueue")


class AsyncMessageQueue:
    """
    In-process async message queue with pub/sub topics.
    
    In production, replace with Redis Pub/Sub or Kafka.
    This implementation provides the same API so the switch is seamless.
    
    Topics:
    - market_data: Raw candle events
    - features: Computed feature vectors
    - signals: Trading signals
    - orders: Order events
    - risk: Risk alerts
    """
    
    def __init__(self):
        self.topics: dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._running = False
    
    def get_queue(self, topic: str) -> asyncio.Queue:
        """Get the queue for a topic."""
        return self.topics[topic]
    
    async def publish(self, topic: str, message: dict):
        """Publish a message to a topic."""
        await self.topics[topic].put(message)
        logger.debug(f"Published to '{topic}': {message.get('type', 'unknown')}")
    
    async def subscribe(self, topic: str, handler: Callable):
        """
        Subscribe to a topic with an async handler function.
        The handler will be called for each message.
        """
        self.subscribers[topic].append(handler)
        logger.info(f"Subscribed handler to topic '{topic}'")
    
    async def start_consuming(self, topic: str):
        """Start consuming messages for a topic, calling all subscribers."""
        self._running = True
        queue = self.topics[topic]
        
        while self._running:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                for handler in self.subscribers[topic]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error on topic '{topic}': {e}")
                queue.task_done()
            except asyncio.TimeoutError:
                continue
    
    def stop(self):
        """Stop all consumer loops."""
        self._running = False
        logger.info("Message queue stopped")


class RedisMessageQueue:
    """
    Redis-backed pub/sub message queue for distributed services.
    Falls back to AsyncMessageQueue if Redis is unavailable.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        self._host = host
        self._port = port
        self._redis = None
        self._pubsub = None
        self._use_redis = False
        self._fallback = AsyncMessageQueue()
        self._listener_tasks: dict[str, asyncio.Task] = {}
        
    @classmethod
    async def create(cls, host: str = "localhost", port: int = 6379):
        instance = cls(host, port)
        try:
            import redis.asyncio as aioredis
            instance._redis = aioredis.Redis(host=host, port=port)
            await instance._redis.ping()
            instance._use_redis = True
            logger.info(f"Redis MQ connected at {host}:{port}")
        except Exception as e:
            instance._use_redis = False
            logger.warning(f"Redis MQ unavailable ({e}). Using in-memory queue.")
        return instance
    
    async def publish(self, topic: str, message: dict):
        """Publish to Redis channel or fallback queue."""
        if self._use_redis:
            try:
                await self._redis.publish(topic, json.dumps(message, default=str))
                return
            except Exception as e:
                logger.error(f"Redis publish error: {e}")
        
        await self._fallback.publish(topic, message)
    
    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to Redis channel or fallback queue."""
        if self._use_redis:
            try:
                if self._pubsub is None:
                    self._pubsub = self._redis.pubsub()
                await self._pubsub.subscribe(topic)
                
                async def _listen():
                    try:
                        async for message in self._pubsub.listen():
                            if message['type'] == 'message':
                                try:
                                    data = json.loads(message['data'])
                                    await handler(data)
                                except Exception as e:
                                    logger.error(f"Error processing message on topic '{topic}': {e}")
                    except asyncio.CancelledError:
                        logger.info(f"Listener for topic '{topic}' cancelled.")
                    except Exception as e:
                        logger.error(f"Redis listen error on topic '{topic}': {e}")
                
                task = asyncio.create_task(_listen())
                self._listener_tasks[topic] = task
                logger.info(f"Subscribed to Redis channel '{topic}'")
                return
            except Exception as e:
                logger.error(f"Redis subscribe error: {e}")
        
        await self._fallback.subscribe(topic, handler)
        
    async def unsubscribe(self, topic: str):
        """Unsubscribe from a topic and cancel its listener task."""
        if self._use_redis and self._pubsub:
            await self._pubsub.unsubscribe(topic)
        if topic in self._listener_tasks:
            self._listener_tasks[topic].cancel()
            try:
                await self._listener_tasks[topic]
            except asyncio.CancelledError:
                pass
            del self._listener_tasks[topic]
    
    def get_queue(self, topic: str) -> asyncio.Queue:
        """Get direct queue access (works with fallback only)."""
        return self._fallback.get_queue(topic)
