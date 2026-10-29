import logging
import time
from contextlib import asynccontextmanager
from app.api import chat, conversations, health
import torch
from fastapi import FastAPI
from app.db.session import engine
from app.api import chat, health
from app.config import settings
from app.ml.generation import ChatGenerator
from app.ml.loader import load_model, load_tokenizer

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)

    t0 = time.perf_counter()
    model = load_model(settings.ckpt_path, settings.device)
    tok = load_tokenizer(settings.tokenizer_path)
    app.state.generator = ChatGenerator(model, tok, settings.device)
    logger.info(
        "Model loaded in %.1fs from %s, torch threads=%d",
        time.perf_counter() - t0, settings.ckpt_path, torch.get_num_threads(),
    )

    yield

    app.state.generator = None
    engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(title="GOOD-NLP chat", lifespan=lifespan)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(conversations.router)