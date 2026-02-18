import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ERPNext
    ERPNEXT_URL = os.getenv("ERPNEXT_URL", "").rstrip("/")

    # AI Provider
    AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # openai, anthropic, ollama

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Ollama
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
