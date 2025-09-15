from django.contrib import admin
from django.urls import path
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from dashboard import api as fastapi_app  # import your FastAPI app

fastapi_as_wsgi = WSGIMiddleware(fastapi_app)

urlpatterns = [
    path("admin/", admin.site.urls),
]
