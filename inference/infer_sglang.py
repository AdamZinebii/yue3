"""
YuE Music Generation Inference using SGLang

This script uses SGLang Engine for faster inference with native CFG (guidance_scale) support.
Replaces the transformers-based approach for improved performance.

Usage:
    python infer_sglang.py --genre_txt genre.txt --lyrics_txt lyrics.txt --output_dir ./output
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))

import re
import random
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from omegaconf import OmegaConf

import sglang as sgl
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched


# =============================================================================
# Custom Logit Processor for Token Blocking
# =============================================================================
class BlockTokenRangeProcessor(CustomLogitProcessor):
    """
    Custom logit processor that blocks specific token ranges.
    Equivalent to the transformers LogitsProcessor used in the original infer.py
    """
    
    def __call__(self, logits, custom_param_list):
        """
        Block tokens in specified ranges by setting their logits to -inf.
        
        Args:
            logits: Tensor of shape (batch_size, vocab_size)
            custom_param_list: List of dicts with 'blocked_ranges' key
        """
        assert logits.shape[0] == len(custom_param_list)
        
        for i, param_dict in enumerate(custom_param_list):
            blocked_ranges = param_dict.get("blocked_ranges", [])
            for start_id, end_id in blocked_ranges:
                logits[i, start_id:end_id] = -float("inf")
        
        return logits


class Stage2LogitProcessor(CustomLogitProcessor):
    """
    Custom logit processor for Stage 2 that blocks specific token ranges.
    """
    
    def __call__(self, logits, custom_param_list):
        assert logits.shape[0] == len(custom_param_list)
        
        for i, param_dict in enumerate(custom_param_list):
            vocab_size = param_dict.get("vocab_size", logits.shape[1])
            # Block range 0-46358 and 53526-vocab_size
            logits[i, 0:46358] = -float("inf")
            logits[i, 53526:vocab_size] = -float("inf")
        
        return logits


# =============================================================================
# Argument Parser
# =============================================================================
parser = argparse.ArgumentParser(description="YuE Music Generation with SGLang")

# Model Configuration
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot",
                    help="Model checkpoint path for Stage 1")
parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general",
                    help="Model checkpoint path for Stage 2")
parser.add_argument("--max_new_tokens", type=int, default=3000,
                    help="Maximum new tokens per generation pass")
parser.add_argument("--repetition_penalty", type=float, default=1.1,
                    help="Repetition penalty (1.0 = no penalty)")
parser.add_argument("--run_n_segments", type=int, default=2,
                    help="Number of segments to process")
parser.add_argument("--stage2_batch_size", type=int, default=4,
                    help="Batch size for Stage 2 inference")

# Prompt Configuration
parser.add_argument("--genre_txt", type=str, required=True,
                    help="Path to genre tags file")
parser.add_argument("--lyrics_txt", type=str, required=True,
                    help="Path to lyrics file")
parser.add_argument("--use_audio_prompt", action="store_true",
                    help="Use audio file as prompt")
parser.add_argument("--audio_prompt_path", type=str, default="",
                    help="Path to audio prompt file")
parser.add_argument("--prompt_start_time", type=float, default=0.0,
                    help="Start time for audio prompt extraction")
parser.add_argument("--prompt_end_time", type=float, default=30.0,
                    help="End time for audio prompt extraction")
parser.add_argument("--use_dual_tracks_prompt", action="store_true",
                    help="Use dual tracks as prompt")
parser.add_argument("--vocal_track_prompt_path", type=str, default="",
                    help="Path to vocal track prompt")
parser.add_argument("--instrumental_track_prompt_path", type=str, default="",
                    help="Path to instrumental track prompt")

# Output Configuration
parser.add_argument("--output_dir", type=str, default="./output",
                    help="Output directory")
parser.add_argument("--keep_intermediate", action="store_true",
                    help="Keep intermediate outputs")
parser.add_argument("--disable_offload_model", action="store_true",
                    help="Don't offload model after Stage 1")
parser.add_argument("--cuda_idx", type=int, default=0,
                    help="CUDA device index")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

# Codec/Vocoder Configuration
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml',
                    help='xcodec config YAML')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth',
                    help='xcodec checkpoint path')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml',
                    help='Vocos config path')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth',
                    help='Vocos vocal decoder weights')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth',
                    help='Vocos instrumental decoder weights')
parser.add_argument('-r', '--rescale', action='store_true',
                    help='Rescale output to avoid clipping')

args = parser.parse_args()

# Validate arguments
if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError("--audio_prompt_path required when --use_audio_prompt is set")
if args.use_dual_tracks_prompt and not (args.vocal_track_prompt_path and args.instrumental_track_prompt_path):
    raise FileNotFoundError("Both --vocal_track_prompt_path and --instrumental_track_prompt_path required when --use_dual_tracks_prompt is set")


# =============================================================================
# Utility Functions
# =============================================================================
def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audio_mono(filepath, sampling_rate=16000):
    """Load audio file and convert to mono"""
    audio, sr = torchaudio.load(filepath)
    audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    """Encode audio using codec model"""
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes


def split_lyrics(lyrics):
    """Split lyrics into structured segments"""
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    """Save audio tensor to file"""
    folder_path = os.path.dirname(path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


# =============================================================================
# Initialize
# =============================================================================
seed_everything(args.seed)

# Setup directories
stage1_output_dir = os.path.join(args.output_dir, "stage1")
stage2_output_dir = os.path.join(args.output_dir, "stage2")
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)

# Setup device
device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")

# Load tokenizer
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")

# Load codec tools
codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)

# Load codec model
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

# =============================================================================
# Stage 1: Generate Music Tokens with SGLang
# =============================================================================
print("Initializing Stage 1 model with SGLang...")

# Initialize SGLang Engine for Stage 1
stage1_engine = sgl.Engine(
    model_path=args.stage1_model,
    dtype="bfloat16",
)

# Load and prepare prompts
with open(args.genre_txt) as f:
    genres = f.read().strip()
with open(args.lyrics_txt) as f:
    lyrics = split_lyrics(f.read())

full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics

# Generation parameters
random_id = uuid.uuid4()
top_p = 0.93
temperature = 1.0
repetition_penalty = args.repetition_penalty

# Special tokens
start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
end_of_segment = mmtokenizer.tokenize('[end_of_segment]')

# Stage 1 generation loop
stage1_output_set = []
run_n_segments = min(args.run_n_segments + 1, len(lyrics))
raw_output_ids = []

# Create the logit processor for Stage 1
stage1_logit_processor = BlockTokenRangeProcessor()

for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage 1 inference...")):
    section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
    guidance_scale = 1.5 if i <= 1 else 1.2
    
    if i == 0:
        continue
    
    # Build prompt IDs
    if i == 1:
        if args.use_dual_tracks_prompt or args.use_audio_prompt:
            if args.use_dual_tracks_prompt:
                vocals_ids = load_audio_mono(args.vocal_track_prompt_path)
                instrumental_ids = load_audio_mono(args.instrumental_track_prompt_path)
                vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
                instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
                vocals_ids = codectool.npy2ids(vocals_ids[0])
                instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
                audio_prompt_codec = ids_segment_interleaved[int(args.prompt_start_time*50*2): int(args.prompt_end_time*50*2)]
                audio_prompt_codec = audio_prompt_codec.tolist()
            elif args.use_audio_prompt:
                audio_prompt = load_audio_mono(args.audio_prompt_path)
                raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                code_ids = codectool.npy2ids(raw_codes[0])
                audio_prompt_codec = code_ids[int(args.prompt_start_time * 50): int(args.prompt_end_time * 50)]
            audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
            sentence_ids = mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
            head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
        else:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
        prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
    else:
        prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

    # Combine with previous output
    if i > 1:
        input_ids = raw_output_ids + prompt_ids
    else:
        input_ids = prompt_ids
    
    # Apply context window limit
    max_context = 16384 - args.max_new_tokens - 1
    if len(input_ids) > max_context:
        print(f'Section {i}: output length {len(input_ids)} exceeding context {max_context}, truncating...')
        input_ids = input_ids[-max_context:]
    
    # Generate with SGLang
    # Note: SGLang uses input_ids directly, we need to decode to text or use token IDs
    # For now, we convert token IDs to a format SGLang can process
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 100,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "stop_token_ids": [mmtokenizer.eoa],
        "guidance_scale": guidance_scale,
        "custom_params": {
            "blocked_ranges": [(0, 32002), (32016, 32017)]
        }
    }
    
    # SGLang generation with input_ids
    output = stage1_engine.generate(
        input_ids=[input_ids],
        sampling_params=sampling_params,
        custom_logit_processor=stage1_logit_processor.to_str(),
    )
    
    # Extract generated token IDs
    output_ids = output[0]["token_ids"] if "token_ids" in output[0] else []
    
    # Append EOA if not present
    if not output_ids or output_ids[-1] != mmtokenizer.eoa:
        output_ids.append(mmtokenizer.eoa)
    
    # Update raw output
    if i > 1:
        raw_output_ids = raw_output_ids + prompt_ids + output_ids
    else:
        raw_output_ids = input_ids + output_ids

# Shutdown Stage 1 engine
if not args.disable_offload_model:
    stage1_engine.shutdown()
    del stage1_engine
    torch.cuda.empty_cache()

# =============================================================================
# Process Stage 1 Output
# =============================================================================
ids = np.array(raw_output_ids)
soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()

if len(soa_idx) != len(eoa_idx):
    raise ValueError(f'Invalid SOA/EOA pairs: {len(soa_idx)} SOA, {len(eoa_idx)} EOA')

vocals = []
instrumentals = []
range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0

for i in range(range_begin, len(soa_idx)):
    codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
    if codec_ids[0] == 32016:
        codec_ids = codec_ids[1:]
    codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
    vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
    vocals.append(vocals_ids)
    instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
    instrumentals.append(instrumentals_ids)

vocals = np.concatenate(vocals, axis=1)
instrumentals = np.concatenate(instrumentals, axis=1)

# Save Stage 1 outputs
vocal_save_path = os.path.join(
    stage1_output_dir,
    f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{args.max_new_tokens}_{random_id}_vtrack".replace('.', '@') + '.npy'
)
inst_save_path = os.path.join(
    stage1_output_dir,
    f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{args.max_new_tokens}_{random_id}_itrack".replace('.', '@') + '.npy'
)

np.save(vocal_save_path, vocals)
np.save(inst_save_path, instrumentals)
stage1_output_set.append(vocal_save_path)
stage1_output_set.append(inst_save_path)

print(f"Stage 1 complete. Saved: {vocal_save_path}, {inst_save_path}")

# =============================================================================
# Stage 2: Refine Music Tokens with SGLang
# =============================================================================
print("\nInitializing Stage 2 model with SGLang...")

stage2_engine = sgl.Engine(
    model_path=args.stage2_model,
    dtype="bfloat16",
)

stage2_logit_processor = Stage2LogitProcessor()


def stage2_generate(engine, prompt, batch_size=16):
    """Generate Stage 2 refined tokens"""
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)
    
    # Prepare prompts for batch processing
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])
        codec_ids = np.concatenate(codec_list, axis=0)
        
        prompt_ids_list = []
        for i in range(batch_size):
            prompt_ids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids[i],
                np.array([mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids_list.append(prompt_ids.tolist())
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids_list = [prompt_ids.tolist()]
    
    # Teacher forcing generation loop
    all_outputs = [[] for _ in range(len(prompt_ids_list))]
    current_prompts = [list(p) for p in prompt_ids_list]
    
    num_frames = codec_ids.shape[1] if batch_size > 1 else len(codec_ids.flatten())
    
    for frames_idx in range(num_frames):
        # Add the codec token for this frame
        for b in range(len(current_prompts)):
            if batch_size > 1:
                cb0 = int(codec_ids[b, frames_idx])
            else:
                cb0 = int(codec_ids.flatten()[frames_idx])
            current_prompts[b].append(cb0)
        
        # Generate 7 tokens
        sampling_params = {
            "min_new_tokens": 7,
            "max_new_tokens": 7,
            "temperature": 0.0,
            "stop_token_ids": [mmtokenizer.eoa],
            "custom_params": {"vocab_size": mmtokenizer.vocab_size}
        }
        
        outputs = engine.generate(
            input_ids=current_prompts,
            sampling_params=sampling_params,
            custom_logit_processor=stage2_logit_processor.to_str(),
        )
        
        # Update prompts and collect outputs
        for b, out in enumerate(outputs):
            new_tokens = out.get("token_ids", [])
            current_prompts[b].extend(new_tokens)
            all_outputs[b].extend(new_tokens)
    
    # Concatenate batch outputs
    if batch_size > 1:
        output = np.concatenate([np.array(o) for o in all_outputs], axis=0)
    else:
        output = np.array(all_outputs[0])
    
    return output


def stage2_inference(engine, stage1_output_set, stage2_output_dir, batch_size=4):
    """Run Stage 2 inference on all Stage 1 outputs"""
    stage2_result = []
    
    for i in tqdm(range(len(stage1_output_set)), desc="Stage 2 inference..."):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} already exists, skipping.')
            stage2_result.append(output_filename)
            continue
        
        # Load prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Process in 6-second segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            output = stage2_generate(engine, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
            
            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)
                current_batch_size = batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(engine, prompt[:, start_idx:end_idx], batch_size=current_batch_size)
                segments.append(segment)
            
            output = np.concatenate(segments, axis=0)
        
        # Process ending
        if output_duration * 50 != prompt.shape[-1]:
            ending = stage2_generate(engine, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        
        output = codectool_stage2.ids2npy(output)
        
        # Fix invalid codes
        fixed_output = copy.deepcopy(output)
        for j, line in enumerate(output):
            for k, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[j, k] = most_frequent
        
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    
    return stage2_result


stage2_result = stage2_inference(stage2_engine, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
print(f"Stage 2 complete: {stage2_result}")

# Shutdown Stage 2 engine
stage2_engine.shutdown()

# =============================================================================
# Audio Reconstruction and Post-processing
# =============================================================================
print("\nReconstructing audio from tokens...")

recons_output_dir = os.path.join(args.output_dir, "recons")
recons_mix_dir = os.path.join(recons_output_dir, 'mix')
os.makedirs(recons_mix_dir, exist_ok=True)

tracks = []
for npy in stage2_result:
    codec_result = np.load(npy)
    with torch.no_grad():
        decoded_waveform = codec_model.decode(
            torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
    tracks.append(save_path)
    save_audio(torch.as_tensor(decoded_waveform), save_path, 16000)

# Mix tracks
recons_mix = None
for inst_path in tracks:
    try:
        if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) and '_itrack' in inst_path:
            vocal_path = inst_path.replace('_itrack', '_vtrack')
            if not os.path.exists(vocal_path):
                continue
            recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
            vocal_stem, sr = sf.read(inst_path)
            instrumental_stem, _ = sf.read(vocal_path)
            mix_stem = (vocal_stem + instrumental_stem) / 1
            sf.write(recons_mix, mix_stem, sr)
    except Exception as e:
        print(f"Mix error: {e}")

# Vocoder upsampling
print("\nUpsampling with vocoder...")
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
os.makedirs(vocoder_mix_dir, exist_ok=True)
os.makedirs(vocoder_stems_dir, exist_ok=True)

instrumental_output = None
vocal_output = None

for npy in stage2_result:
    if '_itrack' in npy:
        instrumental_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'itrack.mp3'),
            args.rescale,
            args,
            inst_decoder,
            codec_model
        )
    else:
        vocal_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
            args.rescale,
            args,
            vocal_decoder,
            codec_model
        )

# Mix vocoder outputs
try:
    if instrumental_output is not None and vocal_output is not None:
        mix_output = instrumental_output + vocal_output
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix) if recons_mix else "mixed.mp3")
        save_audio(mix_output, vocoder_mix, 44100, args.rescale)
        print(f"Created vocoder mix: {vocoder_mix}")
        
        # Post-process
        if recons_mix:
            replace_low_freq_with_energy_matched(
                a_file=recons_mix,
                b_file=vocoder_mix,
                c_file=os.path.join(args.output_dir, os.path.basename(recons_mix)),
                cutoff_freq=5500.0
            )
except RuntimeError as e:
    print(f"Final mix failed: {e}")

print("\n" + "=" * 50)
print("Generation complete!")
print(f"Output directory: {args.output_dir}")
print("=" * 50)
