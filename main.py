from fastapi import FastAPI
from api.routes.reviews_route import router as reviews_router  # 👈 Use underscore


app = FastAPI()

# Include the router from another file
app.include_router(reviews_router)