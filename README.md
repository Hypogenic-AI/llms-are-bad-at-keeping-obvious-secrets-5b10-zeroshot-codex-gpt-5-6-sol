# Can a Plan Keep a Secret?

This repository tests whether an explicit, secret-independent story outline reduces involuntary secret leakage from an LLM. It also asks whether secret identity remains linearly decodable from the model's residual stream during story generation, and includes an exploratory directional ablation.

## Main result

The experiment did **not** recover the required positive leakage anchor. A position-balanced GPT-4.1-mini reader identified secret-conditioned Gemma 3 12B stories at 54.2% in the plain condition (95% pair-bootstrap CI 39.6–68.8%; 24 matched pairs). Outline accuracy was 50.0%; length-matched irrelevant context was 41.7%; and a decoy was 39.6%. The outline-minus-plain difference was −4.2 percentage points (paired randomization p=0.836). Because plain generation was already statistically indistinguishable from chance, these data cannot establish that outlines reduce leakage. The likely design-level explanation is that giving even the plain writer a fixed premise already constrains content choice enough to create a floor.

Across non-decoy conditions, an eight-way cross-premise linear probe also provided no robust evidence that secret identity remained in story-period residual states. The best of 28 inspected layer/quartile cells reached 20.8% versus 12.5% chance (uncorrected p=0.047), but did not survive multiplicity correction; final-quartile accuracy at that layer was 9.7%. Probe confidence did not correlate with behavioral leakage (Spearman rho=0.059, p=0.623).

All 96 secret-conditioned stories obeyed literal nondisclosure. The exploratory layer-24 projection drove detection to 6.3% versus 56.3% for a random direction, but catastrophically degraded the stories (mean quality 1.23 versus 4.02; mean length 98 versus 273 words, often malformed). It is therefore a failed selective intervention, not evidence that a secret feature was removed.

The finished paper is [`paper_draft/main.pdf`](paper_draft/main.pdf), with source in [`paper_draft/main.tex`](paper_draft/main.tex).

## Design

- Writer: `unsloth/gemma-3-12b-it-bnb-4bit`, sampled at temperature 0.8 and top-p 0.95.
- Reader: `openai/gpt-4.1-mini` through OpenRouter, temperature 0.
- 8 secrets × 3 fixed premises × 4 conditions × secret/no-secret = 192 stories.
- Conditions: plain, content outline, approximately length-matched editorial distractor, and deterministic decoy.
- Each of the 96 matched pairs was judged in both presentation orders.
- Primary estimate: pair-level mean of the two order judgments; 95% bootstrap CI over pairs.
- Contrasts: paired sign-flip randomization over the 24 secret/premise blocks.
- Probe: residual states pooled over four story quartiles at seven depths; 32-component PCA plus multinomial logistic regression, with one entire premise held out per fold. Decoys are excluded because their one-to-one mapping to secrets would make decoding tautological.

## Reproducing

Hardware used here was one NVIDIA RTX A6000 (48 GB). Model weights and caches go under `.cache/` and are ignored by git. API keys are read only from environment variables and are never stored.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# Generate the 192-story matrix (resumable).
.venv/bin/python src/generate.py --repeats 1 --batch-size 8

# Judge every pair in both orders (resumable; requires OPENROUTER_KEY).
.venv/bin/python src/judge.py

# Behavioral statistics and figures.
.venv/bin/python src/analyze.py

# Extract and analyze residual-stream features.
.venv/bin/python src/extract_activations.py
.venv/bin/python src/analyze_probe.py

# Exploratory causal intervention and its judgments.
.venv/bin/python src/ablate_batched.py
.venv/bin/python src/judge_ablation.py
.venv/bin/python src/analyze.py

# Compile the paper.
cd paper_draft
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Generation and judging scripts append one JSON object per line and skip completed IDs, so interrupted runs can be resumed. Exact prompts, stimuli, decoy mappings, seeds, model identifiers, raw model text, raw reader decisions, confidence scores, and quality scores are retained.

## Repository layout

- `src/stimuli.py`: fixed secrets, premises, outlines, distractors, and decoys.
- `src/generate.py`: quantized writer generation.
- `src/judge.py`: position-balanced 2AFC reader.
- `src/analyze.py`: behavioral and intervention statistics and figures.
- `src/extract_activations.py`, `src/analyze_probe.py`: residual extraction and cross-premise probe.
- `src/ablate_batched.py`, `src/judge_ablation.py`: exploratory directional removal and evaluation.
- `results/*.jsonl`: raw stories and judgments.
- `results/activations.npz`: pooled residual states (float32; compressed).
- `results/derived/`: analysis tables and diagnostic plot.
- `paper_draft/`: LaTeX, bibliography, figures, and compiled paper.

## Interpretation

This is an informative null result, not evidence that models generally keep secrets or that outlines cannot help. Earlier work found strong leakage when writers freely chose story content. Here, all conditions used a fixed premise. A follow-up should cross both factors—free topic versus fixed premise, with versus without an outline—using multiple writers and readers before making a general claim about planning.
