import os
import django
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from django.core.asgi import get_asgi_application

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dashboard.settings')
django.setup()

# Import your FastAPI app
from dashboard.api import app as fastapi_app

# Optionally add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Django ASGI app
django_asgi_app = get_asgi_application()

# Mount Django under "/" and FastAPI under "/api"
from starlette.applications import Starlette

app = Starlette()
app.mount("/api", fastapi_app)  # FastAPI routes
app.mount("/", django_asgi_app) # Django routes
