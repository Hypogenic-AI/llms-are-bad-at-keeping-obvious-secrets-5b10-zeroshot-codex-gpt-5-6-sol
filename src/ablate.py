#!/usr/bin/env python3
"""Exploratory causal intervention: remove a prompt-derived secret direction at layer 24."""
import argparse, hashlib, json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from generate import prompt, SYSTEM
from stimuli import SECRETS, PREMISES

LAYER=24

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="unsloth/gemma-3-12b-it-bnb-4bit")
    ap.add_argument("--out",default="results/ablations.jsonl"); ap.add_argument("--premises",type=int,default=3); args=ap.parse_args()
    out=Path(args.out); done=set()
    if out.exists():
      for line in open(out):
       try:
        x=json.loads(line); done.add((x["secret"],x["premise"],x["intervention"]))
       except Exception: pass
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(args.model,cache_dir=".cache/huggingface")
    model=AutoModelForCausalLM.from_pretrained(args.model,cache_dir=".cache/huggingface",device_map={"":0},quantization_config=quant,torch_dtype=torch.bfloat16,attn_implementation="sdpa").eval()
    layer=model.model.language_model.layers[LAYER-1]
    for secret in tqdm(SECRETS):
      for pi in range(args.premises):
        ps=prompt(secret,pi,"plain",True); pc=prompt(secret,pi,"plain",False)
        def enc(p):
          c=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
          return c,tok(c,return_tensors="pt").to(model.device)
        chat_s,ins=enc(ps); _,inc=enc(pc)
        with torch.inference_mode():
          hs=model(**ins,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1).hidden_states[LAYER][0,-1].float()
          hc=model(**inc,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1).hidden_states[LAYER][0,-1].float()
        direction=hs-hc; direction=direction/direction.norm()
        rg=torch.Generator(device=model.device).manual_seed(int(hashlib.sha256(f"random-{secret}-{pi}".encode()).hexdigest()[:8],16))
        random_dir=torch.randn(direction.shape,device=model.device,generator=rg); random_dir-=random_dir.dot(direction)*direction; random_dir/=random_dir.norm()
        for name,u in [("secret_direction",direction),("random_direction",random_dir)]:
          if (secret,pi,name) in done: continue
          u=u.to(torch.bfloat16)
          def hook(module,inputs,output):
            h=output[0]; proj=torch.einsum("btd,d->bt",h,u).unsqueeze(-1)
            return (h-proj*u,)+output[1:]
          handle=layer.register_forward_hook(hook)
          seed=int(hashlib.sha256(f"intervention-{secret}-{pi}".encode()).hexdigest()[:8],16); torch.manual_seed(seed)
          with torch.inference_mode(): ids=model.generate(**ins,max_new_tokens=430,do_sample=True,temperature=.8,top_p=.95,repetition_penalty=1.03,pad_token_id=tok.eos_token_id)
          handle.remove(); text=tok.decode(ids[0,ins.input_ids.shape[1]:],skip_special_tokens=True).strip()
          rec={"id":f"{secret}-{pi}-{name}","secret":secret,"premise":pi,"intervention":name,"layer":LAYER,
               "seed":seed,"prompt":ps,"text":text,"words":len(text.split()),"direction_norm":float((hs-hc).norm())}
          with out.open("a") as f:f.write(json.dumps(rec)+"\n")
          done.add((secret,pi,name))

if __name__=="__main__": main()

