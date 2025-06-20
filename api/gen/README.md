# 🎬 Movie Review Generator

A Python utility that pulls real user sentiments from IMDb, fetches factual movie data from OMDb, then prompts a large-language model to write concise, structured film reviews—with an embedded sentiment verdict.

---

## 🚀 Features

- **IMDb sentiment mining**  
  Scans the Stanford IMDb dataset for your title, tallies positive vs. negative mentions.
- **OMDb fact lookup**  
  Fetches director, cast, genre & plot via the OMDb API for factual grounding.
- **Prompt templating**  
  Builds a strict “critic-style” prompt that enforces structure, avoids hallucination.
- **Adaptive sentiment**  
  Falls back to OMDb ratings when IMDb samples are inconclusive.
- **Clean output**  
  Post-processes model text into short, punchy sentences.
- **Built-in sentiment check**  
  Runs the final review through a classifier (POSITIVE/NEGATIVE + confidence).

---

## 📋 Prerequisites

- Python 3.8+
- An LLM checkpoint (e.g. `gpt2`, or any HuggingFace-compatible causal LM)
- OMDb API key
- (Optional) Separate sentiment model checkpoint

---

## ⚙️ Installation

1. Clone this repo
   ```bash
   git clone https://github.com/your-org/movie-review-generator.git
   cd movie-review-generator
   ```

## 📈 Review Generation Steps

1. **Load & Scan IMDb**

   - Pulls the first 1,000 IMDb reviews and searches for mentions of your movie title.
   - Counts up to 5 sentiment-labeled hits (positive vs. negative).

2. **Build a Structured Prompt**

   - Embeds the following factual details:
     - **Director**
     - **Cast**
     - **Genre**
     - **Plot**
   - Includes the calculated sentiment phrase.
   - Follows a strict 4-point format:
     1. **Quick Synopsis**
     2. **Key Highlights**
     3. **Theme & Takeaway** (with sentiment)
     4. **Final Score** (out of 5★)

3. **Generate Review via LLM**

   - Sends the prompt into a HuggingFace text-generation pipeline configured with the model and sampling hyper parameters.

4. **Clean & Validate**

   - Trims extraneous whitespace.
   - Splits output into sentences and keeps the first 3 substantial ones.
   - Ensures at least a fallback sentence if generation fails.

5. **Sentiment Classification**
   - Feeds the final review back into your sentiment-analysis pipeline.
   - Returns a JSON response:
