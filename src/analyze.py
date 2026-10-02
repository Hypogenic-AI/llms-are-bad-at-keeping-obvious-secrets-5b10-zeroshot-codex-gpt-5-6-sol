#!/usr/bin/env python3
"""Compute preregistered aggregate statistics and paper figures/tables."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import seaborn as sns

RNG=np.random.default_rng(20261002)
Path("results/derived").mkdir(parents=True,exist_ok=True); Path("paper_draft/figures").mkdir(parents=True,exist_ok=True)
stories=pd.DataFrame(map(json.loads,open("results/stories.jsonl")))
judges=pd.DataFrame(map(json.loads,open("results/judgments.jsonl")))
judges=judges[judges.choice.isin(["A","B"])].copy()
pairs=judges.groupby(["pair_id","secret","premise","condition"],as_index=False).agg(score=("correct","mean"),n_orders=("correct","size"))

def boot_ci(x,B=20000):
    x=np.asarray(x,float); vals=np.mean(RNG.choice(x,(B,len(x)),replace=True),axis=1)
    return np.quantile(vals,[.025,.975])

rows=[]
for c,g in pairs.groupby("condition"):
    lo,hi=boot_ci(g.score); k=int(judges[judges.condition.eq(c)].correct.sum()); n=len(judges[judges.condition.eq(c)])
    rows.append({"condition":c,"pairs":len(g),"accuracy":g.score.mean(),"ci_lo":lo,"ci_hi":hi,
                 "judge_correct":k,"judge_n":n,"binom_p":binomtest(k,n,.5).pvalue})
summary=pd.DataFrame(rows).sort_values("condition"); summary.to_csv("results/derived/condition_summary.csv",index=False)

wide=pairs.pivot(index=["secret","premise"],columns="condition",values="score")
comparisons=[]
for c in ["outline","distractor","decoy"]:
    d=(wide[c]-wide["plain"]).dropna().to_numpy(); obs=d.mean()
    null=np.mean(RNG.choice([-1,1],(200000,len(d)))*d,axis=1)
    p=(1+(np.abs(null)>=abs(obs)).sum())/(len(null)+1)
    lo,hi=boot_ci(d)
    comparisons.append({"comparison":f"{c}-plain","mean_difference":obs,"ci_lo":lo,"ci_hi":hi,"permutation_p":p,"n":len(d)})
pd.DataFrame(comparisons).to_csv("results/derived/comparisons.csv",index=False)

# Literal compliance and descriptive lengths.
stories["literal_secret"]=stories.apply(lambda r: bool(r.has_secret and r.secret.lower() in r.text.lower()),axis=1)
story_stats=stories.groupby(["condition","has_secret"]).agg(n=("id","size"),mean_words=("words","mean"),sd_words=("words","std"),literal=("literal_secret","sum")).reset_index()
story_stats.to_csv("results/derived/story_stats.csv",index=False)

# Quality is assigned to the underlying secret/control text after undoing order.
q=[]
for _,r in judges.iterrows():
    if r.order=="secret_first": qs,qc=r.quality_a,r.quality_b
    else: qc,qs=r.quality_a,r.quality_b
    q.append({"condition":r.condition,"pair_id":r.pair_id,"secret_quality":qs,"control_quality":qc})
quality=pd.DataFrame(q).groupby(["condition","pair_id"],as_index=False).mean(numeric_only=True)
quality.groupby("condition").agg(secret_quality=("secret_quality","mean"),control_quality=("control_quality","mean")).to_csv("results/derived/quality.csv")

sns.set_theme(style="whitegrid",context="paper",font_scale=1.15)
fig,ax=plt.subplots(figsize=(6.3,3.5)); order=["plain","outline","distractor","decoy"]
s=summary.set_index("condition").loc[order]
ax.errorbar(range(4),s.accuracy,yerr=[s.accuracy-s.ci_lo,s.ci_hi-s.accuracy],fmt="o",capsize=4,color="#2457a6",markersize=7)
ax.axhline(.5,color="black",ls="--",lw=1); ax.set(xticks=range(4),xticklabels=[x.title() for x in order],ylabel="2AFC leakage accuracy",ylim=(.25,1.01),xlabel="Writer condition")
fig.tight_layout(); fig.savefig("paper_draft/figures/leakage.pdf",bbox_inches="tight"); fig.savefig("results/derived/leakage.png",dpi=180,bbox_inches="tight"); plt.close(fig)

# Per-secret plot exposes heterogeneity.
sec=pairs.groupby(["secret","condition"],as_index=False).score.mean()
fig,ax=plt.subplots(figsize=(7.2,4.1)); sns.pointplot(data=sec,x="secret",y="score",hue="condition",hue_order=order,ax=ax,markers=["o","s","^","D"],linestyles="none")
ax.axhline(.5,color="black",ls="--",lw=1); ax.set(ylabel="Mean 2AFC accuracy",xlabel="Secret",ylim=(-.03,1.03)); ax.tick_params(axis="x",rotation=30); ax.legend(title="Condition",ncol=2)
fig.tight_layout(); fig.savefig("paper_draft/figures/by_secret.pdf",bbox_inches="tight"); plt.close(fig)
print(summary.to_string(index=False)); print(pd.DataFrame(comparisons).to_string(index=False)); print(story_stats.to_string(index=False))

# Exploratory interventions, when available.
if Path("results/ablation_judgments.jsonl").exists():
    aj=pd.DataFrame(map(json.loads,open("results/ablation_judgments.jsonl")))
    aj=aj[aj.choice.isin(["A","B"])]
    apair=aj.groupby(["pair_id","secret","premise","intervention"],as_index=False).agg(score=("correct","mean"))
    ar=[]
    for c,g in apair.groupby("intervention"):
        lo,hi=boot_ci(g.score); ar.append({"intervention":c,"pairs":len(g),"accuracy":g.score.mean(),"ci_lo":lo,"ci_hi":hi})
    aw=apair.pivot(index=["secret","premise"],columns="intervention",values="score")
    dd=(aw.secret_direction-aw.random_direction).dropna().to_numpy(); obs=dd.mean()
    null=np.mean(RNG.choice([-1,1],(200000,len(dd)))*dd,axis=1); pp=(1+(np.abs(null)>=abs(obs)).sum())/(len(null)+1)
    pd.DataFrame(ar).to_csv("results/derived/ablation_summary.csv",index=False)
    pd.DataFrame([{"comparison":"secret-random","mean_difference":obs,"permutation_p":pp,"n":len(dd)}]).to_csv("results/derived/ablation_comparison.csv",index=False)
    aq=[]
    for _,r in aj.iterrows():
        qi=r.quality_a if r.order=="secret_first" else r.quality_b
        aq.append({"intervention":r.intervention,"quality":qi})
    pd.DataFrame(aq).groupby("intervention").quality.mean().to_csv("results/derived/ablation_quality.csv")
    print(pd.DataFrame(ar).to_string(index=False)); print("ablation difference",obs,"p",pp)
