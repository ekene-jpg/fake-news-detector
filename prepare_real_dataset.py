"""
Downloads the real Kaggle "Fake and Real News Dataset" and merges it into
data/news_dataset.csv, replacing the synthetic demo data used for initial
pipeline testing.

Requires ONE of these credential setups as environment variables:

  Newer format (single token, starts with "KGAT_"):
    KAGGLE_API_TOKEN

  Older format (from a kaggle.json with "username"/"key" fields):
    KAGGLE_USERNAME
    KAGGLE_KEY

Get either from kaggle.com -> your profile icon -> Settings -> API ->
"Create New Token" (newer format) or "Create Legacy API Key" (older format).

If neither is set, this script does nothing and the existing
data/news_dataset.csv (demo or previously-downloaded real data) is left
untouched — so it's always safe to run.

Usage:
    python data/prepare_real_dataset.py
"""
import os
import subprocess
import pandas as pd

DATASET = "clmentbisaillon/fake-and-real-news-dataset"
RAW_DIR = "data/kaggle_raw"


def main():
    has_new_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_legacy_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if not (has_new_token or has_legacy_creds):
        print("[skip] No Kaggle credentials found (KAGGLE_API_TOKEN, or KAGGLE_USERNAME+KAGGLE_KEY) "
              "— keeping existing data/news_dataset.csv")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading {DATASET} from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR, "--unzip"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[warn] Kaggle download failed ({e}) — keeping existing data/news_dataset.csv")
        return
    except FileNotFoundError:
        print("[warn] `kaggle` command not found — is it in requirements.txt? Keeping existing dataset.")
        return

    fake_path = os.path.join(RAW_DIR, "Fake.csv")
    true_path = os.path.join(RAW_DIR, "True.csv")
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        print("[warn] Expected Fake.csv / True.csv not found after download — keeping existing dataset.")
        return

    fake = pd.read_csv(fake_path)
    fake["label"] = 0  # 0 = Fake
    real = pd.read_csv(true_path)
    real["label"] = 1  # 1 = Real

    df = (
        pd.concat([fake, real], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    df.to_csv("data/news_dataset.csv", index=False)
    print(
        f"Wrote data/news_dataset.csv: {len(df)} rows "
        f"({(df.label == 0).sum()} fake, {(df.label == 1).sum()} real)"
    )


if __name__ == "__main__":
    main()
