import os
import logging
import re
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

@dataclass
class ReviewConfig:
    """Configuration for concise review generation."""
    max_new_tokens: int = 150
    temperature: float = 0.6
    top_k: int = 30
    top_p: float = 0.8
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3

class MovieReviewGenerator:
    """Generator for concise, structured movie reviews (not summaries)."""
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("MODEL_ID")
        self.config = ReviewConfig()
        self._tokenizer = None
        self._model = None
        self._generator = None
        self._imdb_dataset = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._initialize_model()
        return self._tokenizer

    @property
    def generator(self):
        if self._generator is None:
            self._initialize_model()
        return self._generator

    @property
    def imdb_dataset(self):
        if self._imdb_dataset is None:
            self._initialize_dataset()
        return self._imdb_dataset

    def _initialize_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            logger.info(f"Loading model: {self.model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._generator = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _initialize_dataset(self):
        try:
            logger.info("Loading IMDb dataset...")
            self._imdb_dataset = load_dataset("stanfordnlp/imdb", cache_dir=".cache")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")

    def _get_movie_context(self, movie_title: str) -> Dict[str, Any]:
        """Fetch movie facts from OMDb to ensure factual accuracy."""
        context = {"director": "Unknown", "stars": [], "genre": "film", "plot": "story"}
        if OMDB_API_KEY:
            try:
                resp = requests.get(
                    "http://www.omdbapi.com/",
                    params={"t": movie_title, "apikey": OMDB_API_KEY}
                )
                data = resp.json()
                context["director"] = data.get("Director", context["director"])
                cast_list = data.get("Actors", "").split(", ")
                context["stars"] = cast_list if cast_list else context["stars"]
                context["genre"] = data.get("Genre", context["genre"])
                context["plot"] = data.get("Plot", context["plot"])
            except Exception as e:
                logger.warning(f"OMDb lookup failed: {e}")
        return context

    @lru_cache(maxsize=64)
    def _find_relevant_sentiment(self, movie_title: str) -> str:
        title = movie_title.lower().strip()
        patterns = [re.compile(rf"\b{re.escape(title)}\b", re.IGNORECASE)]
        pos, neg = 0, 0
        for i, item in enumerate(self.imdb_dataset['train']):
            if i >= 1000:
                break
            text = item.get('text', '')
            if patterns[0].search(text):
                if item.get('label', 0) == 1:
                    pos += 1
                else:
                    neg += 1
                if pos + neg >= 5:
                    break
        if pos > neg:
            return "generally well-received"
        if neg > pos:
            return "received mixed reactions"
        return "has divided critics"

    def _create_structured_prompt(self, movie_title: str) -> str:
        sentiment = self._find_relevant_sentiment(movie_title)
        context = self._get_movie_context(movie_title)
        cast_str = ", ".join(context['stars']) if context['stars'] else 'Cast information unavailable'

        prompt = (
            f"You are a seasoned film critic. Write a concise, engaging REVIEW of '{movie_title}' "
            f"(not a summary). ONLY use the facts provided below; do NOT invent any other names, roles, or plot points.\n"
            f"Director: {context['director']}\n"
            f"Cast: {cast_str}\n"
            "Follow this format strictly:\n"
            "1. Quick Synopsis (1–2 sentences, no spoilers)\n"
            "2. Key Highlights (acting, direction, visuals in 2–3 sentences)\n"
            f"3. Theme & Takeaway (1–2 sentences; include sentiment: {sentiment})\n"
            "4. Final Score (out of 5★)\n"
            "Keep under ~120 words. Start below exactly as shown (no extra headings):\nReview:"""
        )
        logger.info(f"Structured prompt created: {prompt[:100]}...")
        return prompt

    def _clean_and_validate_review(self, raw_text: str, movie_title: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r'[.!?]+', text)
        kept = []
        for s in sentences:
            s = s.strip()
            if len(s) < 15:
                continue
            kept.append(s[0].upper() + s[1:])
            if len(kept) >= 3:
                break
        if not kept:
            return f"{movie_title} is a compelling film that showcases strong storytelling and memorable performances."
        result = '. '.join(kept)
        return result + ('.' if not result.endswith('.') else '')

    def generate_review(self, movie_title: str) -> str:
        if not movie_title.strip():
            raise ValueError("Movie title cannot be empty")
        logger.info(f"Generating review for '{movie_title}'")
        prompt = self._create_structured_prompt(movie_title)
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            raw = outputs[0]["generated_text"]
            logger.info(f"Raw output: {raw[:80]}...")
            return self._clean_and_validate_review(raw, movie_title)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"{movie_title} is a noteworthy film with engaging performances and solid direction."

# Simple interface
generator = MovieReviewGenerator()
def generateMovieReview(movie_title: str) -> str:
    return generator.generate_review(movie_title)
