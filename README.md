# Sarcasm Detector

Baseline (TF-IDF + LogReg) and transformer fine-tuning scripts for sarcasm
detection, plus a script to pull together public sarcasm datasets.

## 1. Push this to a GitHub repo

```bash
git init
git add .
git commit -m "sarcasm detector setup"
gh repo create sarcasm-detector --private --source=. --push
```

## 2. Launch a Codespace (4-core, 16GB)

Via the GitHub web UI:
- Go to your repo -> Code -> Codespaces -> "..." -> New with options
- Machine type: pick the 16-core / 64GB option (shows as available machine
  types based on your GitHub plan; may be labeled "Large" or similar)

Or via GitHub CLI:
```bash
gh codespace create --repo YOUR_USERNAME/sarcasm-detector --machine largePremiumLinux
```
(Run `gh codespace create --repo YOUR_USERNAME/sarcasm-detector` without
`--machine` first to see the exact machine-type names available to your
account/org — they vary by plan.)

The `.devcontainer/devcontainer.json` requests 4 cores / 16GB and
auto-installs everything in `requirements.txt` on creation.

Note: this machine size is CPU-only. Fine-tuning a transformer on CPU works
but is slow (expect hours, not minutes, for a few epochs on tens of
thousands of rows). If you want this to run in under an hour, add a GPU
Codespace machine type (if available on your plan) instead — the code
doesn't need any changes, `transformers`/`torch` will use the GPU
automatically if present.

## 3. Inside the Codespace, get the data

```bash
python download_data.py --out reviews_sarcasm.csv --sarc_sample 50000
```

This pulls and combines:
- News Headlines (Onion vs HuffPost) — clean, formal, ~28k rows
- SARC (Reddit, self-annotated) — sampled to 50k rows by default
- Multi-Sarcasm (Reddit + chatbot dialogue)

None of these are customer-review text specifically — there's no clean
public review-sarcasm dataset — so this gives you a general sarcasm
detector to start from. For real accuracy on customer reviews, plan to
hand-label a few hundred of your own reviews later and mix them in (or
fine-tune this model further on them).

## 4. Run the baseline

```bash
python baseline_sarcasm.py --data reviews_sarcasm.csv
```

Fast, gives you a sanity-check F1 score in under a minute.

## 5. Fine-tune the transformer

```bash
python finetune_sarcasm.py --data reviews_sarcasm.csv --epochs 3 --batch_size 32
```

With 16 cores / 64GB and ~100k rows, expect this to take a while on CPU
(check `htop`/`nproc` to confirm all cores are in use). Reduce
`--sarc_sample` in step 3 if you want faster iteration first.
