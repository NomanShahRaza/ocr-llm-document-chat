from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
UPLOAD_DIR = DATA_DIR / 'uploads'
RESULT_DIR = DATA_DIR / 'results'
EXPORT_DIR = Path(os.getenv('EXPORT_DIR', BASE_DIR / 'results'))
RAG_DIR = Path(os.getenv('RAG_DIR', DATA_DIR / 'rag'))

MODEL_VERSION = os.getenv('PADDLEOCR_VL_VERSION', 'v1.6')
ENGINE = os.getenv('PADDLEOCR_ENGINE', 'paddle')
ENGINES = {
    'paddle': f'PaddleOCR-VL-{MODEL_VERSION.removeprefix("v")}',
    'xiaomi': 'Xiaomi-OCR-0',
}
DEVICE = os.getenv('PADDLEOCR_DEVICE', '') or None
ENABLE_ORIENTATION = os.getenv('ENABLE_ORIENTATION', 'true').lower() == 'true'
ENABLE_UNWARPING = os.getenv('ENABLE_UNWARPING', 'true').lower() == 'true'
_LOCAL_XIAOMI = BASE_DIR / 'models' / 'Xiaomi-OCR-0'
XIAOMI_MODEL = os.getenv('XIAOMI_OCR_MODEL') or (
    str(_LOCAL_XIAOMI) if (_LOCAL_XIAOMI / 'model.safetensors').is_file() else 'xiaomi-research/Xiaomi-OCR-0'
)
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '100'))

# RAG: BGE-M3 dense + BM25 sparse hybrid search
BGE_M3_MODEL = os.getenv('BGE_M3_MODEL', 'BAAI/bge-m3')
RAG_DEVICE = os.getenv('RAG_DEVICE', 'cpu')  # keep GPU free for OCR by default
RAG_CHUNK_CHARS = int(os.getenv('RAG_CHUNK_CHARS', '900'))
RAG_CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '120'))
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '8'))
RAG_RRF_K = int(os.getenv('RAG_RRF_K', '60'))
RAG_AUTO_INDEX = os.getenv('RAG_AUTO_INDEX', 'true').lower() == 'true'

# Answer synthesis: any OpenAI-compatible chat endpoint (OpenAI, Ollama, LM Studio, vLLM).
# Leave LLM_MODEL empty to use the built-in extractive fallback.
LLM_BASE_URL = os.getenv('LLM_BASE_URL', '') or None
LLM_API_KEY = os.getenv('LLM_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', '')
LLM_TIMEOUT = float(os.getenv('LLM_TIMEOUT', '120'))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '700'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.2'))
LLM_CONTEXT_HITS = int(os.getenv('LLM_CONTEXT_HITS', '8'))
LLM_HISTORY_TURNS = int(os.getenv('LLM_HISTORY_TURNS', '6'))

ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

for d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, EXPORT_DIR, RAG_DIR):
    d.mkdir(parents=True, exist_ok=True)
