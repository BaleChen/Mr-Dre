"""
Refine a feedback file into localized edit plans (see mrdre/feedback_refine.py).
"""

import logging

from dotenv import load_dotenv

# Load .env before importing mrdre modules: the disk caches read MCP_CACHE_DIR when they are imported
load_dotenv()

from mrdre.feedback_refine import main  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
