"""HTTP API:  POST /v1/search, POST /v1/ask, GET /health.

Run:  python -m coderag serve --index indexes/flask
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .index import MODES, CodeIndex
from .qa import ExtractiveGenerator, GeneratorError, answer_question


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    mode: str = "auto"


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


def create_app(index: CodeIndex, generator=None) -> Starlette:
    generator = generator or ExtractiveGenerator()

    async def _parse(request: Request, model):
        try:
            return model.model_validate(await request.json()), None
        except ValueError as exc:
            if isinstance(exc, ValidationError):
                return None, JSONResponse({"error": "validation_error", "details": exc.errors(include_url=False,
                                           include_context=False)}, status_code=422)
            return None, JSONResponse({"error": "invalid_json"}, status_code=400)

    async def search(request: Request) -> JSONResponse:
        body, err = await _parse(request, SearchRequest)
        if err:
            return err
        if body.mode not in MODES:
            return JSONResponse({"error": "validation_error", "details": f"mode must be one of {list(MODES)}"},
                                status_code=422)
        try:
            hits = await run_in_threadpool(index.search, body.query, body.k, body.mode)
        except ValueError as exc:
            return JSONResponse({"error": "bad_request", "details": str(exc)}, status_code=400)
        return JSONResponse({"query": body.query, "mode": body.mode, "results": [
            {"rank": h.rank, "location": h.chunk.location, "kind": h.chunk.kind, "name": h.chunk.name,
             "score": round(h.score, 4), "signature": h.chunk.signature} for h in hits]})

    async def ask(request: Request) -> JSONResponse:
        body, err = await _parse(request, AskRequest)
        if err:
            return err
        try:
            result = await run_in_threadpool(answer_question, index, body.question, generator, body.k)
        except GeneratorError as exc:
            return JSONResponse({"error": "generator_unavailable", "details": str(exc)}, status_code=502)
        return JSONResponse(result)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "chunks": len(index.chunks), "backend": index.backend.name,
                             "reranker": index.reranker is not None, "generator": generator.name,
                             "cache": index.cache_info()})

    return Starlette(routes=[Route("/v1/search", search, methods=["POST"]), Route("/v1/ask", ask, methods=["POST"]),
                             Route("/health", health, methods=["GET"])])
