# Lazy import: avoids pulling in sqlalchemy/fastapi when only schemas are needed (e.g., unit tests).
try:
    from .routes import router
    __all__ = ["router"]
except ImportError:
    pass
