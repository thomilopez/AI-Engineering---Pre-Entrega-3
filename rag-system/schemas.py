from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    answer: str = Field(
        description="La respuesta generada por el modelo basándose únicamente en el contexto proporcionado."
    )
    references: list[str] = Field(
        description="Lista de referencias (nombres de archivo o IDs de chunk) utilizadas para generar la respuesta."
    )
    grounded: bool = Field(
        description="Indica si la respuesta fue generada con base en el contexto (true) o si el modelo indicó no tener acceso a la información (false)."
    )
