"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import documents, queries, agent_routes, canvas

app = FastAPI(
    title="Financial RAG API",
    description="PDF ingestion, RAG queries, agentic research, and canvas persistence.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(queries.router,   prefix="/queries",   tags=["queries"])
app.include_router(agent_routes.router, prefix="/agent",  tags=["agent"])
app.include_router(canvas.router,    prefix="/canvas",    tags=["canvas"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Financial RAG API. See /docs for endpoints."}
