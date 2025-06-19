from fastapi import APIRouter
from api.gen.generator import generateMovieReview 
from pydantic import BaseModel

router = APIRouter(prefix="/reviews", tags=["Reviews"])
class ReviewRequest(BaseModel):
    movie_title: str

@router.post("/generate")
def review(request: ReviewRequest):
    # validate the request
    if not request.movie_title or not request.movie_title.strip():
        return {"error": "Movie title cannot be empty", "is_success": False}
    return {"is_success": True, "data": {"review": generateMovieReview(request.movie_title)}}
