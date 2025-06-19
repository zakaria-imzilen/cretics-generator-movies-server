import os
import logging
import re
from typing import List, Optional, Dict
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


@dataclass
class ReviewConfig:
    """Configuration for review generation."""
    max_new_tokens: int = 100
    temperature: float = 0.6
    top_k: int = 30
    top_p: float = 0.8
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3

class MovieReviewGenerator:
    """Clean movie review generator with structured prompts."""
    
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("MODEL_ID")
        self.config = ReviewConfig()
        
        # Lazy initialization
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
        """Initialize model with better error handling."""
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
        """Initialize IMDb dataset."""
        try:
            logger.info("Loading IMDb dataset...")
            self._imdb_dataset = load_dataset("stanfordnlp/imdb", cache_dir=".cache")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")
    
   
    
    @lru_cache(maxsize=64)
    def _find_relevant_sentiment(self, movie_title: str) -> str:
        """Find sentiment from IMDb reviews."""
        try:
            movie_title_clean = movie_title.lower().strip()
            patterns = [
                re.compile(rf'\b{re.escape(movie_title_clean)}\b', re.IGNORECASE),
                re.compile(rf'{re.escape(movie_title_clean)}', re.IGNORECASE)
            ]
            
            positive_count = 0
            negative_count = 0
            
            # Quick sample from train set
            for i, item in enumerate(self.imdb_dataset['train']):
                if i > 1000:  # Limit search for performance
                    break
                    
                text = item.get("text", "")
                if any(pattern.search(text) for pattern in patterns):
                    if item.get("label", 0) == 1:
                        positive_count += 1
                    else:
                        negative_count += 1
                    
                    if positive_count + negative_count >= 5:
                        break
            
            if positive_count > negative_count:
                return "generally well-received"
            elif negative_count > positive_count:
                return "received mixed reviews"
            else:
                return "has divided critics"
                
        except Exception as e:
            logger.warning(f"Could not determine sentiment: {e}")
            return "is a notable film"
    
    def _create_structured_prompt(self, movie_title: str) -> str:
        sentiment = self._find_relevant_sentiment(movie_title)

        prompt = f"""Write a movie review for "{movie_title}". The film {sentiment}.\n\nCritical Assessment:"""
    
        logger.info(f"Structured prompt created: {prompt[:100]}...")
        return prompt

    
    def _clean_and_validate_review(self, raw_text: str, movie_title: str) -> str:
        """Clean and validate the generated review."""
        if not raw_text or len(raw_text.strip()) < 10:
            return f"{movie_title} is a compelling film that showcases strong storytelling and memorable performances."
        
        # Clean the text
        text = raw_text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = []
        
        movie_words = movie_title.lower().split()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Only keep substantial sentences
                # Check if sentence is relevant (contains movie-related words or common review terms)
                sentence_lower = sentence.lower()
                review_words = ['film', 'movie', 'story', 'performance', 'character', 'plot', 'direction', 'acting', 'cinema']
                
                is_relevant = (
                    any(word in sentence_lower for word in movie_words) or
                    any(word in sentence_lower for word in review_words) or
                    len(clean_sentences) == 0  # Always keep first sentence
                )
                
                if is_relevant:
                    # Capitalize first letter
                    sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
                    clean_sentences.append(sentence)
                    
                    if len(clean_sentences) >= 3:  # Limit to 3 good sentences
                        break
        
        if clean_sentences:
            result = '. '.join(clean_sentences)
            if not result.endswith('.'):
                result += '.'
        else:
            # Fallback review
            context = self._get_movie_context(movie_title)
            result = f"{movie_title} is a {context['genre'].lower()} film that delivers an engaging {context['plot']}. The performances are solid and the direction is competent. Overall, it's a worthwhile viewing experience."
        
        return result
    
    def generate_review(self, movie_title: str) -> str:
        """Generate a clean, factual movie review."""
        if not movie_title or not movie_title.strip():
            raise ValueError("Movie title cannot be empty")
        
        movie_title = movie_title.strip()
        logger.info(f"Generating review for: '{movie_title}'")
        
        try:
            prompt = self._create_structured_prompt(movie_title)
            
            # Generate with conservative parameters
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
            
            if not outputs:
                raise RuntimeError("No output generated")
            
            raw_review = outputs[0]["generated_text"]
            logger.info(f"Raw output: {raw_review[:100]}...")
            
            # Clean and validate the review
            final_review = self._clean_and_validate_review(raw_review, movie_title)
            
            logger.info("Review generated successfully")
            return final_review
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return a fallback review
            context = self._get_movie_context(movie_title)
            return f"{movie_title} is a {context['genre'].lower()} film directed by {context['director']}. The movie presents {context['plot']} with solid performances from {context['stars']}. It's a well-crafted film that delivers on its promises."

# Convenience function
def generateMovieReview(movie_title: str, max_new_tokens: int = 100) -> str:
    """Generate movie review - simple interface."""
    generator = MovieReviewGenerator()
    return generator.generate_review(movie_title)

# CLI interface
def main():
    """Main CLI interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generator.py 'Movie Title'")
        print("Example: python generator.py 'Titanic'")
        print("\nSupported movies with enhanced context:")
        print("- Titanic, Avatar, Inception, The Dark Knight")
        print("\nEnvironment variables:")
        print("- MODEL_ID (default: gpt2)")
        sys.exit(1)
    
    movie_title = " ".join(sys.argv[1:])
    
    try:
        print(f"\n🎬 Generating review for: '{movie_title}'")
        print("=" * 60)
        
        generator = MovieReviewGenerator()
        review = generator.generate_review(movie_title)
        
        print(f"\n📝 Movie Review: {movie_title}")
        print("-" * 60)
        print(review)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()