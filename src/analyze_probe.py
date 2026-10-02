#!/usr/bin/env python3
"""Cross-premise secret-identity decoding and leakage correlation."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, binomtest
import matplotlib.pyplot as plt
import seaborn as sns

d=np.load("results/activations.npz"); X=d["features"].astype("float32"); layers=d["layers"]
meta=pd.DataFrame([json.loads(x) for x in d["meta"]])
# The decoy is deterministically paired with the secret and would make identity
# decoding tautological. Exclude it from the mechanistic probe.
keep=meta.condition.ne("decoy").to_numpy(); meta=meta.loc[keep].reset_index(drop=True); X=X[keep]
labels=sorted(meta.secret.unique()); y=meta.secret.map({x:i for i,x in enumerate(labels)}).to_numpy()
records=[]; probs=np.zeros((len(meta),len(layers),4))
for li,layer in enumerate(layers):
 for b in range(4):
  pred=np.empty(len(meta),int); truep=np.empty(len(meta))
  for held in sorted(meta.premise.unique()):
   tr=meta.premise.ne(held).to_numpy(); te=~tr
   # Dimensionality reduction is fit inside each training fold; it regularizes
   # the p >> n setting and makes the preregistered linear decoder tractable.
   clf=make_pipeline(StandardScaler(),PCA(n_components=32,svd_solver="randomized",random_state=20261002),LogisticRegression(C=.1,max_iter=1000,solver="lbfgs"))
   clf.fit(X[tr,li,b],y[tr]); pred[te]=clf.predict(X[te,li,b]); truep[te]=clf.predict_proba(X[te,li,b])[np.arange(te.sum()),y[te]]
  probs[:,li,b]=truep
  for cond in sorted(meta.condition.unique()):
   ix=meta.condition.eq(cond).to_numpy(); records.append({"layer":int(layer),"quartile":b+1,"condition":cond,"accuracy":(pred[ix]==y[ix]).mean(),"n":int(ix.sum())})
recdf=pd.DataFrame(records); recdf.to_csv("results/derived/probe_accuracy.csv",index=False)
overall=(recdf.assign(correct=lambda z:z.accuracy*z.n).groupby(["layer","quartile"],as_index=False).agg(correct=("correct","sum"),n=("n","sum")))
overall["accuracy"]=overall.correct/overall.n
overall["binom_p_uncorrected"]=overall.apply(lambda r:binomtest(int(round(r.correct)),int(r.n),1/8).pvalue,axis=1)
overall.to_csv("results/derived/probe_overall.csv",index=False)

# Primary representation strength: held-out true-class probability at the best aggregate layer, averaged across story quartiles.
agg=[]
for li,layer in enumerate(layers): agg.append((probs[:,li].mean(),li,int(layer)))
_,best_i,best_layer=max(agg)
meta["probe_strength"]=probs[:,best_i].mean(1)
jud=pd.DataFrame(map(json.loads,open("results/judgments.jsonl")))
pair=jud[jud.choice.isin(["A","B"])].groupby("pair_id",as_index=False).correct.mean().rename(columns={"correct":"leakage"})
meta["pair_id"]=meta.apply(lambda r:f"{r['secret']}-{r['premise']}-{r['condition']}-{r['repeat']}",axis=1)
m=meta.merge(pair,on="pair_id"); rho,p=spearmanr(m.probe_strength,m.leakage)
pd.DataFrame([{"best_layer":best_layer,"spearman_rho":rho,"p":p,"n":len(m),"mean_true_class_probability":m.probe_strength.mean()}]).to_csv("results/derived/probe_correlation.csv",index=False)
m.to_csv("results/derived/probe_story_scores.csv",index=False)

sns.set_theme(style="whitegrid",context="paper",font_scale=1.1)
df=recdf; fig,axs=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
for ax,q in zip(axs,[1,4]):
 z=df[df.quartile.eq(q)]; sns.lineplot(data=z,x="layer",y="accuracy",hue="condition",marker="o",ax=ax)
 ax.axhline(1/8,color="black",ls="--",lw=1); ax.set(title=f"Story quartile {q}",ylabel="Secret identity accuracy",ylim=(0,1))
axs[1].legend(title="Condition",fontsize=8); axs[0].get_legend().remove(); fig.tight_layout(); fig.savefig("paper_draft/figures/probe.pdf",bbox_inches="tight"); plt.close(fig)
print("best layer",best_layer,"rho",rho,"p",p); print(df.to_string(index=False))
