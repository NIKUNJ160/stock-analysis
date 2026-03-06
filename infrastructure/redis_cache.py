import json
from typing import Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("Infrastructure.Cache")


class RedisCache:
    """
    Redis-backed cache for market data, features, and signals.
    
    Falls back to in-memory dict if Redis is unavailable,
    making it safe to use without a running Redis server.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._redis = None
        self._fallback: dict = {}  # In-memory fallback
        self._fallback_ttl: dict = {}
        self._use_redis = False
        
        try:
            import redis
            self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self._redis.ping()
            self._use_redis = True
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Using in-memory cache fallback.")
    
    def set(self, key: str, value, ttl: int = None) -> bool:
        """Store a value with optional TTL."""
        ttl = ttl or self.default_ttl
        serialized = json.dumps(value, default=str)
        
        if self._use_redis:
            try:
                self._redis.setex(key, ttl, serialized)
                return True
            except Exception as e:
                logger.error(f"Redis SET error: {e}")
        
        self._fallback[key] = serialized
        self._fallback_ttl[key] = time.time() + ttl if ttl else None
        return True
    
    def get(self, key: str) -> Optional[any]:
        """Retrieve a value by key."""
        if self._use_redis:
            try:
                val = self._redis.get(key)
                return json.loads(val) if val else None
            except Exception as e:
                logger.error(f"Redis GET error: {e}")
        
        if key in self._fallback_ttl and self._fallback_ttl[key]:
            if time.time() > self._fallback_ttl[key]:
                self._fallback.pop(key, None)
                self._fallback_ttl.pop(key, None)
                return None
            
        val = self._fallback.get(key)
        return json.loads(val) if val else None
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        if self._use_redis:
            try:
                self._redis.delete(key)
            except Exception as e:
                logger.error(f"Redis DELETE error for {key}: {e}")
        
        self._fallback.pop(key, None)
        self._fallback_ttl.pop(key, None)
        return True
    
    def set_candle_buffer(self, symbol: str, candles: list[dict], max_candles: int = 200):
        """Store recent candle buffer for a symbol."""
        key = f"candles:{symbol}"
        if len(candles) > max_candles:
            candles = candles[-max_candles:]
        self.set(key, candles, ttl=86400)
    
    def get_candle_buffer(self, symbol: str) -> list[dict]:
        """Retrieve candle buffer for a symbol."""
        return self.get(f"candles:{symbol}") or []
    
    def set_latest_signal(self, symbol: str, signal: dict):
        """Store the latest signal for a symbol."""
        self.set(f"signal:{symbol}", signal, ttl=3600)
    
    def get_latest_signal(self, symbol: str) -> Optional[dict]:
        """Get latest signal for a symbol."""
        return self.get(f"signal:{symbol}")
    
    def set_features(self, symbol: str, features: dict):
        """Cache latest computed features."""
        self.set(f"features:{symbol}", features, ttl=600)
    
    def get_features(self, symbol: str) -> Optional[dict]:
        """Get cached features."""
        return self.get(f"features:{symbol}")
    
    def get_all_signals(self) -> dict:
        """Get latest signals for all symbols."""
        if self._use_redis:
            try:
                result = {}
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor=cursor, match="signal:*", count=100)
                    for key in keys:
                        symbol = key.split(":", 1)[1]
                        result[symbol] = self.get(key)
                    if cursor == 0:
                        break
                return result
            except Exception as e:
                logger.error(f"Redis SCAN error in get_all_signals: {e}")
        
        result = {}
        for key, val in self._fallback.items():
            if key.startswith("signal:"):
                symbol = key.split(":", 1)[1]
                result[symbol] = json.loads(val)
        return result
    
    def flush(self):
        """Clear all cached data."""
        if self._use_redis:
            try:
                prefixes = ["candles:*", "signal:*", "features:*"]
                for prefix in prefixes:
                    cursor = 0
                    while True:
                        cursor, keys = self._redis.scan(cursor=cursor, match=prefix, count=500)
                        if keys:
                            self._redis.delete(*keys)
                        if cursor == 0:
                            break
            except Exception as e:
                logger.error(f"Redis flush error: {e}")
        self._fallback.clear()
        logger.info("Cache flushed")
