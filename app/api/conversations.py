from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Conversation
from app.db.session import get_db
from app.schemas.conversation import ConversationDetail, ConversationOut

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    stmt = select(Conversation).order_by(Conversation.id.desc()).limit(50)
    return db.scalars(stmt).all()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id, options=[selectinload(Conversation.messages)])
    if conv is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return conv


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    db.delete(conv)
    db.commit()