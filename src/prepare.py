import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split


def prepare_data(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path).drop_duplicates()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    prepare_data(sys.argv[1], sys.argv[2])