"""
Qwen3-TTS Voice Cloning with Prompt Caching

This script demonstrates voice cloning with prompt reuse:
- Creates voice prompt once from reference audio+text (expensive)
- Saves prompt to disk for reuse across program runs
- Generates multiple texts with same voice using cached prompt (fast)

To regenerate prompt: delete prompt_cache/{voice}.pt
"""
import torch
import soundfile as sf
import warnings
from pathlib import Path
from dataclasses import asdict
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

root = Path(__file__).resolve().parent
model_path = (root / "models" / "Qwen3-TTS-12Hz-1.7B-Base").resolve()

model = Qwen3TTSModel.from_pretrained(
    str(model_path),
    device_map="mps",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)

# Extract model identifiers for cache filename
# This ensures prompts are model-specific (different sizes/tokenizers = different prompts)
model_size = model.model.tts_model_size  # e.g., "1b7" or "0b6"
tokenizer_type = model.model.tokenizer_type  # e.g., "qwen3_tts_tokenizer_12hz"
model_type = model.model.tts_model_type  # e.g., "base"

# voice = "cole1"
# voice = "cole2"
# voice = "obama1"
# voice = "spock1"
voice = "jackie1"

ref_stem = root / "reference_voices" / voice
prompt_cache_dir = root / "prompt_cache"
prompt_cache_dir.mkdir(exist_ok=True)
# Include model size and tokenizer type in cache filename to avoid cross-model reuse
prompt_cache_path = prompt_cache_dir / f"{voice}_{tokenizer_type}_{model_size}_{model_type}.pt"


def save_voice_prompt(prompt_items, path: Path):
    """Save voice clone prompt items to disk."""
    payload = {
        "items": [asdict(it) for it in prompt_items],
    }
    torch.save(payload, path)
    print(f"Saved voice prompt to: {path}")


def load_voice_prompt(path: Path) -> list[VoiceClonePromptItem]:
    """Load voice clone prompt items from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError("Invalid prompt file format")
    
    items = []
    for d in payload["items"]:
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Missing ref_spk_embedding")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text", None),
            )
        )
    return items


# Load cached prompt or create new one
# Cache filename includes model size/tokenizer to prevent cross-model reuse
print(f"Model: {tokenizer_type} | Size: {model_size} | Type: {model_type}")
print(f"Cache file: {prompt_cache_path.name}")

if prompt_cache_path.exists():
    print(f"Loading cached voice prompt from: {prompt_cache_path}")
    prompt_items = load_voice_prompt(prompt_cache_path)
else:
    print(f"Creating new voice prompt for: {voice}")
    with open(ref_stem.with_suffix(".txt"), "rb") as f:
        ref_text = f.read().decode("utf-8")
    
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(ref_stem.with_suffix(".wav")),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    save_voice_prompt(prompt_items, prompt_cache_path)

texts = [
    "Today will be sunny with a high of 25 degrees Celsius and a light breeze. Expect scattered showers throughout the day and a chance of thunderstorms in the evening. Cloudy in the morning with clear skies expected by afternoon. Highs around 20 degrees. It's going to be windy and chilly with temperatures reaching only 15 degrees Celsius. A beautiful, warm day ahead with highs around 28 degrees and no chance of rain.",
]

output_dir = root / "output_audio_wavs"
output_dir.mkdir(exist_ok=True)
base_name = f"output-{voice}"

for text in texts:
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=prompt_items,  # Reuse cached prompt!
    )
    
    # Find next available filename to avoid overwriting
    out_path = output_dir / f"{base_name}.wav"
    if out_path.exists():
        counter = 1
        while True:
            out_path = output_dir / f"{base_name}_{counter}.wav"
            if not out_path.exists():
                break
            counter += 1
    
    sf.write(out_path, wavs[0], sr)
    print(f"Saved: {out_path}")

# Note: To regenerate the prompt (e.g., if reference audio changed), delete:
#   prompt_cache/{voice}_{tokenizer_type}_{model_size}_{model_type}.pt
# Example: prompt_cache/cole1_qwen3_tts_tokenizer_12hz_1b7_base.pt