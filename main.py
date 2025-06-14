from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Movie Review Generator API"}
