from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.reviews_route import router as reviews_router  # 👈 Use underscore


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router from another file
app.include_router(reviews_router)