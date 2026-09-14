import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret_key: str
    access_token_minutes: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://diploma_app:change_me_local@localhost:5432/disease_risk",
        ),
        jwt_secret_key=os.getenv(
            "JWT_SECRET_KEY", "change-this-development-secret-with-32-plus-bytes"
        ),
        access_token_minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "60")),
    )
