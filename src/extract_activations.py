#!/usr/bin/env python3
"""Extract mean residual-stream states over story quartiles."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LAYERS=[0,8,16,24,32,40,48]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="unsloth/gemma-3-12b-it-bnb-4bit")
    ap.add_argument("--stories",default="results/stories.jsonl"); ap.add_argument("--out",default="results/activations.npz"); args=ap.parse_args()
    rows=[json.loads(x) for x in open(args.stories) if json.loads(x)["has_secret"]]
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(args.model,cache_dir=".cache/huggingface")
    model=AutoModelForCausalLM.from_pretrained(args.model,cache_dir=".cache/huggingface",device_map={"":0},quantization_config=quant,torch_dtype=torch.bfloat16,attn_implementation="sdpa").eval()
    from generate import SYSTEM
    feats=[]; meta=[]
    for r in tqdm(rows):
        prefix=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":r["prompt"]}],tokenize=False,add_generation_prompt=True)
        pids=tok(prefix,return_tensors="pt",add_special_tokens=False).input_ids
        full=tok(prefix+r["text"],return_tensors="pt",add_special_tokens=False).to(model.device)
        start=pids.shape[1]; end=full.input_ids.shape[1]
        cuts=np.linspace(start,end,5,dtype=int)
        with torch.inference_mode():
            out=model(**full,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
        arr=[]
        for li in LAYERS:
            h=out.hidden_states[li][0]
            arr.append(torch.stack([h[cuts[b]:cuts[b+1]].float().mean(0) for b in range(4)]).cpu().numpy())
        # Gemma residual magnitudes can exceed IEEE float16 range; retain float32.
        feats.append(np.stack(arr).astype("float32")); meta.append({k:r[k] for k in ["id","secret","premise","condition","repeat"]})
        del out,full
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,features=np.stack(feats),layers=np.array(LAYERS),meta=np.array([json.dumps(x) for x in meta]))

if __name__=="__main__": main()
