"""
Offline-mode utility for running entirely on locally downloaded Hugging
Face model files -- for environments where direct access to
huggingface.co is restricted (e.g. corporate networks).

One-time setup (do this once, on a machine that DOES have Hub access, or
via whatever your company's approved artifact-download process is):

    from transformers import (
        WhisperModel, WhisperFeatureExtractor, WhisperProcessor,
        WhisperForConditionalGeneration, AutoModel, AutoTokenizer,
    )

    WhisperModel.from_pretrained("openai/whisper-small").save_pretrained("/models/whisper-small")
    WhisperFeatureExtractor.from_pretrained("openai/whisper-small").save_pretrained("/models/whisper-small")
    WhisperProcessor.from_pretrained("openai/whisper-small").save_pretrained("/models/whisper-small")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").save_pretrained("/models/whisper-small")

    # AutoModel/AutoTokenizer work for BERT, DistilBERT, or any other
    # AutoModel-compatible text encoder -- swap the repo id as needed.
    AutoModel.from_pretrained("distilbert-base-uncased").save_pretrained("/models/distilbert-base-uncased")
    AutoTokenizer.from_pretrained("distilbert-base-uncased").save_pretrained("/models/distilbert-base-uncased")

(All four Whisper classes can safely save_pretrained into the SAME folder
-- they write different, non-overlapping files: config.json, model
weights, preprocessor_config.json, tokenizer files, generation_config.json,
etc. One "/models/whisper-small" directory ends up self-contained.)

Then, in every script that loads these models:

    from hf_offline_utils import enable_offline_mode
    enable_offline_mode()          # call this BEFORE importing transformers

    ...

    cfg = FSCConfig(
        whisper_model_name="/models/whisper-small",   # local dir, not a repo id
        bert_model_name="/models/distilbert-base-uncased",  # BERT, DistilBERT, etc. all work
    )

Every `.from_pretrained(...)` call in this codebase has been updated to
also pass `local_files_only=True` explicitly, as a second safety net on
top of the environment variables -- so even if enable_offline_mode() is
called late or skipped, individual calls still refuse to reach the
network and instead raise a clear local `OSError` telling you the local
path/cache is missing files, rather than silently trying huggingface.co.
"""

import os


def enable_offline_mode():
    """
    Forces every transformers / huggingface_hub call in this process to
    skip the network entirely and only look at local files / local cache.
    Call this ONCE, as early as possible in your script -- ideally before
    `import transformers` -- since some Hub-related state is read at
    import time.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # also prevents `datasets`-based tooling (if you use it downstream)
    # from trying to reach the Hub
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
