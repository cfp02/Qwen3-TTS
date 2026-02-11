"""
Test: Using saved VoiceClonePromptItem with CustomVoice model for instruction control

This combines:
- Voice cloning from Base model (your custom voice)
- Instruction control from CustomVoice model (style/emotion control)
"""

import torch
import soundfile as sf
from pathlib import Path
from dataclasses import asdict
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

root = Path(__file__).resolve().parent

# Load CustomVoice model (supports instructions)
custom_voice_model = Qwen3TTSModel.from_pretrained(
    str(root / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    device_map="mps",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)

# Load Base model to create/load the voice prompt
base_model = Qwen3TTSModel.from_pretrained(
    str(root / "models" / "Qwen3-TTS-12Hz-1.7B-Base"),
    device_map="mps",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)

voice = "cole1"
ref_stem = root / "reference_voices" / voice
prompt_cache_dir = root / "prompt_cache"
prompt_cache_path = prompt_cache_dir / f"{voice}_qwen3_tts_tokenizer_12hz_1b7_base.pt"


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


# Load the saved voice prompt
if not prompt_cache_path.exists():
    print(f"Error: Prompt cache not found: {prompt_cache_path}")
    print("Please run main.py first to create the voice prompt.")
    exit(1)

print(f"Loading voice prompt from: {prompt_cache_path}")
prompt_items = load_voice_prompt(prompt_cache_path)

# Convert to dict format for model.generate()
voice_clone_prompt_dict = custom_voice_model._prompt_items_to_voice_clone_prompt(prompt_items)

# Test: Generate with different instructions using your cloned voice!
texts_with_instructions = [
    # ("These are the voyages of the Starship Enterprise. Its continuing mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no one has gone before.", ""),  # No instruction
    ("These are the voyages of the Starship Enterprise. Its continuing mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no one has gone before. Throughout our journey, we have encountered countless wonders and faced innumerable challenges. Each discovery brings us closer to understanding the vast universe that surrounds us, while every obstacle teaches us more about ourselves and our place in the cosmos. The pursuit of knowledge knows no bounds, and our commitment to exploration remains unwavering as we venture into the unknown depths of space, seeking answers to questions we have yet to ask and finding beauty in places we never imagined existed.", "Speak quickly and calmly, with a serious tone."),
    # ("These are the voyages of the Starship Enterprise. Its continuing mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no one has gone before.", "Speak slowly and calmly, with a serious tone."),
    # ("These are the voyages of the Starship Enterprise. Its continuing mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no one has gone before.", "Speak in a whisper, very quietly and mysteriously."),
]

output_dir = root / "output_audio_wavs"
output_dir.mkdir(exist_ok=True)

for i, (text, instruct) in enumerate(texts_with_instructions):
    print(f"\nGenerating {i+1}/{len(texts_with_instructions)}")
    print(f"Text: {text[:50]}...")
    print(f"Instruct: {instruct if instruct else '(none)'}")
    
    # Call model.generate() directly to use voice_clone_prompt with instructions
    # The wrapper doesn't expose voice_clone_prompt, so we bypass it
    input_ids = custom_voice_model._tokenize_texts([custom_voice_model._build_assistant_text(text)])
    instruct_ids = [custom_voice_model._tokenize_texts([custom_voice_model._build_instruct_text(instruct)])[0]] if instruct else [None]
    
    # Build ref_ids from prompt items if using ICL mode (needed for voice_clone_prompt)
    ref_ids = None
    if prompt_items[0].icl_mode and prompt_items[0].ref_text:
        # Extract ref_text from prompt items and build ref_ids
        ref_texts_for_ids = [item.ref_text for item in prompt_items]
        ref_ids = []
        for rt in ref_texts_for_ids:
            if rt is None or rt == "":
                ref_ids.append(None)
            else:
                ref_tok = custom_voice_model._tokenize_texts([custom_voice_model._build_ref_text(rt)])[0]
                ref_ids.append(ref_tok)
    
    talker_codes_list, _ = custom_voice_model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        ref_ids=ref_ids,  # Required for ICL mode with voice_clone_prompt
        languages=["English"],
        speakers=[None],  # Will be overridden by voice_clone_prompt
        voice_clone_prompt=voice_clone_prompt_dict,
        non_streaming_mode=True,
    )
    
    # Decode audio codes to waveform
    wavs, sr = custom_voice_model.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
    
    instruct_suffix = f"_instruct_{i}" if instruct else "_no_instruct"
    out_path = output_dir / f"customvoice_clone_{voice}{instruct_suffix}_3.wav"
    sf.write(out_path, wavs[0], sr)
    print(f"Saved: {out_path}")

print("\nDone! Check output_audio_wavs/ for results.")
print("You should hear the same cloned voice with different styles/emotions!")
