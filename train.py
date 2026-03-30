import os

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None or HF_TOKEN.strip() == "":
    raise ValueError("HF_TOKEN is missing.")
