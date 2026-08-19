# Datasets — supports combining multiple sources automatically

`model/train_model.py` **auto-detects every supported dataset** placed in
this folder and merges them into one larger, more diverse training set —
no code changes needed. Using more than one source is genuinely better:
each dataset has its own writing style/topic bias, so combining them makes
the model less likely to just memorize one outlet's quirks.

Currently supported, and safe to use in any combination:

| Dataset | Size | Files needed | Kaggle link |
|---|---|---|---|
| **ISOT** (recommended baseline) | ~45,000 articles | `data/Fake.csv` + `data/True.csv` | https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset |
| **WELFake** (recommended add-on) | ~72,000 articles | `data/WELFake_Dataset.csv` | https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification |
| **LIAR** (optional, different style — short political statements) | ~12,800 statements | `data/liar/train.tsv` (+ `test.tsv`, `valid.tsv`) | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |

## Recommended: use ISOT + WELFake together (~117,000 articles)

1. **ISOT:** Download from the Kaggle link above (free account needed).
   Unzip and place both files directly here:
   ```
   data/Fake.csv
   data/True.csv
   ```
2. **WELFake:** Download from its Kaggle link above. Unzip and place the
   CSV here (whatever it's named — the loader checks a few common names):
   ```
   data/WELFake_Dataset.csv
   ```
3. Run training from the project root:
   ```
   python model/train_model.py
   ```
   You'll see output like:
   ```
   Loaded ISOT Fake and Real News Dataset: 44898 articles
   Loaded WELFake Dataset: 71537 articles
   Combined 2 datasets into one training set.
   Removed 1204 duplicate articles across sources.
   ```
   The combined, de-duplicated set trains automatically — nothing else to configure.

## Adding LIAR too (optional — different style of data)

LIAR is short political statements (not full articles) with 6-way
truthfulness labels, which this project collapses to binary REAL/FAKE
(`true`/`mostly-true`/`half-true` → REAL, `false`/`barely-true`/`pants-fire`
→ FAKE). It's a genuinely different distribution from ISOT/WELFake, so it
adds diversity rather than just more of the same:

1. Download and unzip from the link above.
2. Place the `.tsv` files here:
   ```
   data/liar/train.tsv
   data/liar/test.tsv    (optional)
   data/liar/valid.tsv   (optional)
   ```
3. Re-run `python model/train_model.py` — it'll be picked up automatically
   alongside whatever else is present.

## Don't want to download anything right now?

No problem — `generate_sample_data.py` in this folder creates a smaller,
synthetic-but-realistic `sample_data.csv` (2,000 balanced articles) so the
**entire project runs end-to-end immediately**, no Kaggle account needed.
`train_model.py` automatically falls back to this file only if none of the
real datasets above are found.

```
python data/generate_sample_data.py
python model/train_model.py
```

Swap in any of the real datasets any time later — no code changes needed,
just drop the files in this folder (in any combination) and re-run training.

## Other datasets you could add (need a small loader tweak)

- **FakeNewsNet** — https://github.com/KaiDMML/FakeNewsNet (images + social
  context, more complex format)
- **Kaggle "Fake News" competition dataset** — https://www.kaggle.com/competitions/fake-news/data
- **NELA-GT** — large research-grade dataset (700k+ articles), heavier to work with

These use different column formats. If you want one of them added as an
auto-detected source like ISOT/WELFake/LIAR, a new `_load_xxx()` function
in `model/train_model.py` following the same pattern will do it.
