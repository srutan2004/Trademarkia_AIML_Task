from fastapi import FastAPI
from pydantic import BaseModel

from cache.query_engine import QueryEngine


app = FastAPI(
    title="Semantic Search API",
    description="Semantic cache powered search system",
    version="1.0"
)


# Initialize engine once
engine = QueryEngine()


# Request model
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# POST /query
# -----------------------------
@app.post("/query")
def query_endpoint(request: QueryRequest):

    response = engine.query(request.query)

    return response


# -----------------------------
# GET /cache/stats
# -----------------------------
@app.get("/cache/stats")
def cache_stats():

    return engine.cache.stats()


# -----------------------------
# DELETE /cache
# -----------------------------
@app.delete("/cache")
def clear_cache():

    engine.cache.clear()

    return {
        "message": "Cache cleared successfully"
    }