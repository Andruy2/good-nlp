from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    ckpt_path: Path = Path("models/ckpt_sft.pt")
    tokenizer_path: Path = Path("models/tokenizer.json")
    device: str = "cpu"
    torch_threads: int | None = None
    log_level: str = "INFO"
    database_url: str


settings = Settings()