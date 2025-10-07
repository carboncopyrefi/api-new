from starlette.middleware.base import BaseHTTPMiddleware
from django.db import close_old_connections

class DjangoDBMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Close any old/stale DB connections before handling the request
        close_old_connections()
        response = await call_next(request)
        # Close connections again after the request (optional but clean)
        close_old_connections()
        return response