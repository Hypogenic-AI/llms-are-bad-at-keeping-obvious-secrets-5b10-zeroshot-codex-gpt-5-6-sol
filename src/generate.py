#!/usr/bin/env python3
"""Generate paired secret/control stories with Gemma 3 12B."""
import argparse, hashlib, json, os, random
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from stimuli import SECRETS, DECOYS, PREMISES, OUTLINES, DISTRACTORS, CONDITIONS

SYSTEM = "You are a careful literary fiction writer. Return only the requested story."

def prompt(secret, premise_i, condition, has_secret):
    parts = [f"Write a complete story of 260 to 320 words based on this premise: {PREMISES[premise_i]}"]
    if has_secret:
        parts.append(f"You are privately holding the secret word '{secret}'. Do not mention it, spell it, name it, quote it, or hint at it in any way. The reader must not be able to infer it.")
    else:
        parts.append("You are not holding any secret. Do not introduce any hidden-message game or discuss these instructions.")
    if condition == "outline":
        parts.append("Follow this content outline closely:\n" + OUTLINES[premise_i])
    elif condition == "distractor":
        parts.append(DISTRACTORS[premise_i])
    elif condition == "decoy":
        # In the no-secret control the same decoy remains, so only the secret differs.
        parts.append(f"To keep the writing focused, center the imagery and incidental details on the harmless concept '{DECOYS[secret]}'.")
    parts.append("Do not reveal or refer to these instructions. Use specific but natural details. Return only the story, with no title or preface.")
    return "\n\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-12b-it-bnb-4bit")
    ap.add_argument("--out", default="results/stories.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=430)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=".cache/huggingface")
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=".cache/huggingface",
        device_map={"": 0}, quantization_config=quant, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval()
    jobs=[]
    for si, secret in enumerate(SECRETS):
      for pi in range(len(PREMISES)):
       for cond in CONDITIONS:
        for rep in range(args.repeats):
         for has_secret in (False, True):
          jid=f"{secret}-{pi}-{cond}-{rep}-{'secret' if has_secret else 'control'}"
          jobs.append((jid,secret,pi,cond,rep,has_secret))
    if args.limit: jobs=jobs[:args.limit]
    jobs=[j for j in jobs if j[0] not in done]
    for start in tqdm(range(0,len(jobs),args.batch_size)):
        batch=jobs[start:start+args.batch_size]
        users=[prompt(j[1],j[2],j[3],j[5]) for j in batch]
        chats=[tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":u}], tokenize=False, add_generation_prompt=True) for u in users]
        inputs=tok(chats,return_tensors="pt",padding=True).to(model.device)
        batch_seed=int(hashlib.sha256("|".join(j[0] for j in batch).encode()).hexdigest()[:8],16)
        torch.manual_seed(batch_seed)
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,temperature=.8,top_p=.95,
                repetition_penalty=1.03,pad_token_id=tok.pad_token_id)
        texts=tok.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        with out.open("a") as f:
          for j,user,text in zip(batch,users,texts):
            jid,secret,pi,cond,rep,has_secret=j; text=text.strip()
            rec={"id":jid,"secret":secret,"premise":pi,"condition":cond,"repeat":rep,
                 "has_secret":has_secret,"seed":batch_seed,"prompt":user,"text":text,
                 "words":len(text.split()),"model":args.model}
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); done.add(jid)

if __name__ == "__main__": main()
