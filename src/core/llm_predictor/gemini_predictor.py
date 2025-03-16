import os
from google import genai
from google.genai import types
from src.core.llm_predictor.llm_predictor import LLMPredictor   
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiPredictor(LLMPredictor):
    """Predictor using Google's Gemini API."""
    
    def __init__(self, model: str):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
        )
        return response.text.strip()