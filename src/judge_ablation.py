#!/usr/bin/env python3
"""Judge intervention stories against their pre-generated no-secret controls."""
import json, random
from pathlib import Path
from judge import call, parse, PROMPT

out=Path("results/ablation_judgments.jsonl"); done=set()
if out.exists():
 for line in open(out):
  try:
   x=json.loads(line); done.add((x["pair_id"],x["order"]))
  except Exception:pass
stories=[json.loads(x) for x in open("results/stories.jsonl")]
controls={(x["secret"],x["premise"]):x for x in stories if not x["has_secret"] and x["condition"]=="plain"}
ab=[json.loads(x) for x in open("results/ablations.jsonl")]
jobs=[]
for x in ab:
 pid=x["id"]
 for order in ["secret_first","secret_second"]:jobs.append((pid,x,order))
random.Random(20261003).shuffle(jobs)
for n,(pid,x,order) in enumerate(jobs,1):
 if (pid,order) in done:continue
 c=controls[(x["secret"],x["premise"])]
 a,b=(x,c) if order=="secret_first" else (c,x)
 ans=parse(call("openai/gpt-4.1-mini",PROMPT.format(secret=x["secret"],a=a["text"],b=b["text"])))
 ans.update({"pair_id":pid,"secret":x["secret"],"premise":x["premise"],"intervention":x["intervention"],"order":order,
             "correct":ans.get("choice")== ("A" if order=="secret_first" else "B"),"model":"openai/gpt-4.1-mini"})
 with out.open("a") as f:f.write(json.dumps(ans)+"\n")
 print(n,len(jobs),pid,order,ans.get("choice"),flush=True)
