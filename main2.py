import torch
import soundfile as sf
import warnings
from pathlib import Path
from qwen_tts import Qwen3TTSModel

root = Path(__file__).resolve().parent
model_path = (root / "models" / "Qwen3-TTS-12Hz-1.7B-Base").resolve()

model = Qwen3TTSModel.from_pretrained(
    str(model_path),
    device_map="mps",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)

# voice = "cole1"
voice = "obama1"
# voice = "spock1"

ref_stem = root / "reference_voices" / voice

with open(ref_stem.with_suffix(".txt"), "rb") as f:
    ref_text = f.read().decode("utf-8")

wavs, sr = model.generate_voice_clone(
    text="These are the voyages of the Starship Enterprise. Its continuing mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no one has gone before.",
    language="English",
    ref_audio=str(ref_stem.with_suffix(".wav")),
    ref_text=ref_text,
)
# Find next available filename to avoid overwriting
output_dir = root / "output_audio_wavs"
output_dir.mkdir(exist_ok=True)
base_name = f"output-{voice}"
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