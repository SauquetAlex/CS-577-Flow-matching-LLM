"""Flow matching in the last-layer hidden space of Qwen3.5."""

from dotenv import find_dotenv, load_dotenv

# Load local settings (e.g. FLOWLM_DEVICE, HF_TOKEN) from the nearest .env,
# searching upward from the working directory. Variables already set in the
# environment take precedence over the file.
load_dotenv(find_dotenv(usecwd=True), override=False)
