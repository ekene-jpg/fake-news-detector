"""
DEMO DATA GENERATOR — NOT THE REAL DATASET.

Chapter 3 specifies training on the Kaggle "Fake and Real News" dataset
(44,898 total articles: 23,481 fake / 21,417 real). That dataset requires
downloading from Kaggle (needs a free Kaggle account + API token) and
cannot be fetched from this environment (no internet access here).

This script builds a small, clearly-synthetic dataset with the SAME
COLUMN SHAPE (title, text, subject, date, label) purely so the training
pipeline in train_model.py can be demonstrated end-to-end right now.

BEFORE FINAL SUBMISSION: replace data/Fake.csv and data/True.csv (or a
combined data/news_dataset.csv) with the real Kaggle files. See README.md
"Using the real dataset" section for the exact steps.
"""
import pandas as pd
import random

random.seed(42)

fake_templates = [
    "SHOCKING: {who} secretly {action} and mainstream media won't tell you",
    "You won't believe what {who} did to {target} — doctors are furious",
    "BREAKING: {who} caught {action}, internet in chaos over leaked video",
    "Scientists HATE this one weird trick {who} used to {action}",
    "EXPOSED: The truth about {who} that {target} don't want you to know",
    "{who} ADMITS to {action} in shocking late-night confession",
    "Share before they delete this: {who} {action} and got away with it",
    "Miracle cure for {target} discovered by {who}, Big Pharma silent",
]
real_templates = [
    "{who} announced new policy on {target} following council vote",
    "Officials confirm {who} will {action} starting next quarter",
    "Report: {who} data shows steady change in {target} over five years",
    "{who} spokesperson said the {target} measure passed 5-2 on Tuesday",
    "Study published in peer-reviewed journal examines {target} trends",
    "{who} released quarterly earnings, citing growth in {target} sector",
    "Local authorities completed {action} project, according to city records",
    "{who} confirmed {action} after review by an independent committee",
]
whos = ["the mayor", "a tech company", "researchers", "the health ministry", "a celebrity",
        "the school board", "a local hospital", "government officials", "the central bank", "a university team"]
actions = ["hiding the results", "changing the policy", "funding the study", "recalling the product",
           "meeting with investors", "releasing the data", "approving the plan", "auditing the program"]
targets = ["the economy", "public health", "your children", "climate policy", "the housing market",
           "vaccine safety", "local schools", "the water supply", "national security", "small businesses"]

def gen(templates, n, label, subject):
    rows = []
    for i in range(n):
        t = random.choice(templates)
        title = t.format(who=random.choice(whos), action=random.choice(actions), target=random.choice(targets))
        body = (title + ". " + " ".join(random.choice(templates).format(
            who=random.choice(whos), action=random.choice(actions), target=random.choice(targets)
        ) for _ in range(3)))
        rows.append({
            "title": title,
            "text": body,
            "subject": subject,
            "date": f"202{random.randint(2,5)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "label": label
        })
    return rows

fake_rows = gen(fake_templates, 400, 0, "tabloid")   # 0 = Fake
real_rows = gen(real_templates, 400, 1, "news")       # 1 = Real

df = pd.DataFrame(fake_rows + real_rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/news_dataset.csv", index=False)
print(f"Demo dataset written: {len(df)} rows ({(df.label==0).sum()} fake, {(df.label==1).sum()} real)")
print("Reminder: this is SYNTHETIC demo data. Swap in the real Kaggle dataset before final submission.")
