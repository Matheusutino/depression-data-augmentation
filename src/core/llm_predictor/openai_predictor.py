import os
from openai import OpenAI
from src.core.llm_predictor.llm_predictor import LLMPredictor 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIPredictor(LLMPredictor):
    """Predictor using OpenAI's API."""
    
    def __init__(self, model: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key = self.api_key)
        self.model = model  # Model name passed as a parameter

    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content.strip()