"""Local state directory for the research backbone."""

import os

mem0_dir = os.environ.get("MEM0_DIR", os.path.join(os.getcwd(), ".mempursuit"))
os.makedirs(mem0_dir, exist_ok=True)
