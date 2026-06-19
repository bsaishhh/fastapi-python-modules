from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        if not settings.mongo_uri:
            raise RuntimeError("MONGO_URI is not configured")
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_mongo_client()[settings.mongo_db_name]


async def close_mongo_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
