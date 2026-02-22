from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://lotto:lotto123@localhost:5432/lotto_analyzer"
    code_version: str = "v1.0.0"
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,*"

    class Config:
        env_file = ".env"


settings = Settings()
