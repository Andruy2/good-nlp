import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.ml.generation import ChatGenerator, GenerationParams, PromptTooLongError
from app.schemas.chat import ChatRequest, ChatResponse
from sqlalchemy.orm import Session

from app.db.models import Conversation, Message
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def get_generator(request: Request) -> ChatGenerator:
    gen = getattr(request.app.state, "generator", None)
    if gen is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Модель не загружена")
    return gen


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    gen: ChatGenerator = Depends(get_generator),
    db: Session = Depends(get_db),
) -> ChatResponse:
    conv = None
    if req.conversation_id is not None:
        conv = db.get(Conversation, req.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
    db.commit()  # завершаем транзакцию чтения: не держим соединение, пока идёт генерация

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

    if conv is None:
        conv = Conversation(title=req.question[:50])
        db.add(conv)
    db.add_all([
        Message(conversation=conv, role="user", content=req.question),
        Message(conversation=conv, role="assistant", content=res.text, tokens=res.new_tokens),
    ])
    db.commit()

    logger.info("chat: conv=%d prompt=%d new=%d time=%dms",
                conv.id, res.prompt_tokens, res.new_tokens, elapsed_ms)
    return ChatResponse(
        conversation_id=conv.id,
        answer=res.text,
        prompt_tokens=res.prompt_tokens,
        new_tokens=res.new_tokens,
        elapsed_ms=elapsed_ms,
    )