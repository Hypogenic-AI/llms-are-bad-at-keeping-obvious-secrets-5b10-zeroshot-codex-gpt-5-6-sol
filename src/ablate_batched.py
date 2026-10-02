#!/usr/bin/env python3
"""Batched, resumable version of the layer-24 directional intervention."""
import hashlib, json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from generate import prompt, SYSTEM
from stimuli import SECRETS

MODEL="unsloth/gemma-3-12b-it-bnb-4bit"; LAYER=24; OUT=Path("results/ablations.jsonl")
done=set()
if OUT.exists():
 for line in open(OUT):
  try:
   x=json.loads(line); done.add((x["secret"],x["premise"],x["intervention"]))
  except Exception:pass
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(MODEL,cache_dir=".cache/huggingface"); tok.padding_side="left"
if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
model=AutoModelForCausalLM.from_pretrained(MODEL,cache_dir=".cache/huggingface",device_map={"":0},quantization_config=quant,torch_dtype=torch.bfloat16,attn_implementation="sdpa").eval()
layer=model.model.language_model.layers[LAYER-1]

def chat(p):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)

# Compute a direction once for every pair with unfinished work.
info={}
for secret in SECRETS:
 for pi in range(3):
  if all((secret,pi,k) in done for k in ["secret_direction","random_direction"]):continue
  ps,pc=prompt(secret,pi,"plain",True),prompt(secret,pi,"plain",False)
  ins=tok(chat(ps),return_tensors="pt").to(model.device); inc=tok(chat(pc),return_tensors="pt").to(model.device)
  with torch.inference_mode():
   hs=model(**ins,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1).hidden_states[LAYER][0,-1].float()
   hc=model(**inc,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1).hidden_states[LAYER][0,-1].float()
  u=hs-hc; norm=float(u.norm()); u=u/u.norm()
  rg=torch.Generator(device=model.device).manual_seed(int(hashlib.sha256(f"random-{secret}-{pi}".encode()).hexdigest()[:8],16))
  ru=torch.randn(u.shape,device=model.device,generator=rg); ru-=ru.dot(u)*u; ru/=ru.norm()
  info[(secret,pi)]={"prompt":ps,"chat":chat(ps),"secret_direction":u,"random_direction":ru,"norm":norm}

for kind in ["secret_direction","random_direction"]:
 jobs=[(s,p,info[(s,p)]) for (s,p) in info if (s,p,kind) not in done]
 for st in tqdm(range(0,len(jobs),8),desc=kind):
  b=jobs[st:st+8]; inputs=tok([x[2]["chat"] for x in b],return_tensors="pt",padding=True).to(model.device)
  U=torch.stack([x[2][kind] for x in b]).to(torch.bfloat16)
  def hook(module,args,output):
   h=output[0]; proj=torch.einsum("btd,bd->bt",h,U).unsqueeze(-1); return (h-proj*U[:,None,:],)+output[1:]
  handle=layer.register_forward_hook(hook)
  batch_seed=int(hashlib.sha256((kind+"|"+"|".join(f"{x[0]}-{x[1]}" for x in b)).encode()).hexdigest()[:8],16); torch.manual_seed(batch_seed)
  with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=430,do_sample=True,temperature=.8,top_p=.95,repetition_penalty=1.03,pad_token_id=tok.pad_token_id)
  handle.remove(); texts=tok.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
  with OUT.open("a") as f:
   for (secret,pi,z),text in zip(b,texts):
    text=text.strip(); rec={"id":f"{secret}-{pi}-{kind}","secret":secret,"premise":pi,"intervention":kind,"layer":LAYER,"seed":batch_seed,"prompt":z["prompt"],"text":text,"words":len(text.split()),"direction_norm":z["norm"]}
    f.write(json.dumps(rec)+"\n")
