from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=100, ge=1, le=200)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    conversation_id: int 
    answer: str
    prompt_tokens: int
    new_tokens: int
    elapsed_ms: int