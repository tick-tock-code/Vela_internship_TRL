# data/split.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/vcbench_final_public.csv")

print("=== Dataset overview ===")
print(f"Total rows: {len(df)}")
print(f"Success rate: {df['success'].mean():.1%}")
print(f"Columns: {list(df.columns)}")
print("\nSuccess distribution:")
print(df['success'].value_counts())
print(df['success'].value_counts(normalize=True).round(3))

train, val = train_test_split(
    df, test_size=0.2, stratify=df['success'], random_state=42
)
train.to_csv("data/public_train.csv", index=False)
val.to_csv("data/public_val.csv", index=False)

print(f"\nTrain: {len(train)} rows, {train['success'].mean():.1%} positive")
print(f"Val:   {len(val)} rows, {val['success'].mean():.1%} positive")
print("Split saved. Do not re-run this script.")
