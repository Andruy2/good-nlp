import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.ml.generation import ChatGenerator, GenerationParams, PromptTooLongError
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def get_generator(request: Request) -> ChatGenerator:
    gen = getattr(request.app.state, "generator", None)
    if gen is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Модель не загружена")
    return gen


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, gen: ChatGenerator = Depends(get_generator)) -> ChatResponse:
    params = GenerationParams(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    t0 = time.perf_counter()
    try:
        res = gen.generate(req.question, params)
    except PromptTooLongError as e:
        raise HTTPException(status_code=422, detail=str(e))
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    logger.info("chat: prompt=%d new=%d time=%dms", res.prompt_tokens, res.new_tokens, elapsed_ms)
    return ChatResponse(
        answer=res.text,
        prompt_tokens=res.prompt_tokens,
        new_tokens=res.new_tokens,
        elapsed_ms=elapsed_ms,
    )