"""HTTP routers for telegram-fetcher.

Each router is a thin FastAPI ``APIRouter`` that delegates to the application
services stored on ``request.app.state``.
"""
from .auth import router as auth_router
from .channels import router as channels_router
from .download import router as download_router
from .media import router as media_router

__all__ = [
    "auth_router",
    "channels_router",
    "download_router",
    "media_router",
]
