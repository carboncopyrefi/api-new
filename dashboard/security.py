from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from .models import APIKey

from django.db import connections, OperationalError, InterfaceError, close_old_connections

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)):
    key = api_key_header
    if not key:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Missing API key")

    # Helper to get or refresh the API key
    def fetch_api_key():
        return APIKey.objects.get(key=key, active=True)

    try:
        return fetch_api_key()
    except APIKey.DoesNotExist:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or inactive API key")
    except (OperationalError, InterfaceError):
        # Handles "server disconnected" or "lost connection" errors gracefully
        close_old_connections()  # drop any stale DB connections

        # Optionally force reconnect to ensure connection is ready
        try:
            for conn in connections.all():
                conn.connect()
        except Exception:
            pass  # even if reconnect fails here, the retry may still succeed

        # Retry once after reconnect
        try:
            return fetch_api_key()
        except Exception as e:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"Database connection error: {str(e)}",
            )
