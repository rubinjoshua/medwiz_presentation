from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vectorstores"

# Ensure common directories exist (safe if already created)
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Ollama model configuration
# You can change these if you want to experiment with different local models.
OLLAMA_CHAT_MODEL = "llama3.2"  # small, fast model suitable for M2 16GB
OLLAMA_EMBED_MODEL = "llama3.2"  # reuse same model for embeddings via Ollama

# Vector store collection names
SIG_COLLECTION_NAME = "sig_examples"
MED_KB_COLLECTION_NAME = "medical_knowledge"

# Retrieval settings
SIG_K = 3
MED_K = 3

# Presentation settings
SLEEP_TRANSLATION_STAGE = 2.0  # seconds between major log steps in translation
SLEEP_VALIDATION_STAGE = 2.0   # seconds between major log steps in validation
DISPLAY_REFRESH_SECONDS = 1.0  # refresh interval for right-hand display
