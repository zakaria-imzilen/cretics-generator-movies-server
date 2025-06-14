# 🎬 Movie Review Generator API (FastAPI)

This backend generates creative movie reviews using HuggingFace GPT-2 and serves them via a FastAPI REST API.

# 2⃣ (create and) activate an isolated environment

# MacOS
python -m venv .venv
# (Windows ⇒ .venv\Scripts\activate)

source .venv/bin/activate

# 3⃣ install the dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4⃣ launch the FastAPI server with auto-reload
uvicorn main:app --reload --port 8000
