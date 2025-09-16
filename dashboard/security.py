from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from .models import APIKey

API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    key = api_key_header
    if not key:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Missing API key")

    try:
        api_key_obj = APIKey.objects.get(key=key, active=True)
    except APIKey.DoesNotExist:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or inactive API key")

    return api_key_obj