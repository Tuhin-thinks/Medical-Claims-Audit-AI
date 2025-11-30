import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from typing_extensions import Self


class LLM:
    """Singleton LLM class to handle the instance of OpenAI model"""

    _instance: Self | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            load_dotenv()
            cls._instance = super(LLM, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "llm"):
            _openai_key = os.getenv("OPENAI_API_KEY")
            assert _openai_key is not None and isinstance(_openai_key, str), (
                "OPENAI_API_KEY environment variable is not set"
            )
            self.llm = OpenAI(model="gpt-4o", temperature=0.7, api_key=_openai_key)
