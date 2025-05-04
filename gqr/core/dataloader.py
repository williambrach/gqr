import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

DATASET_SIZE = 15_000
TRAIN_SPLIT = 0.2
DEV_SIZE = 1_000

SEED = 42

label2domain = {
    0: "law",
    1: "finance",
    2: "healthcare",
    3: "ood",
}

domain2label = {value : key for key, value in label2domain.items()}

class DataLoader:
    """Handles dataset loading for the GQRBench package."""

    @staticmethod
    def load_train_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the training dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - train: DataFrame containing training examples
                - eval: DataFrame containing evaluation examples
        """
        law_dataset = load_dataset("dim/law_stackexchange_prompts")
        finance_dataset = load_dataset("4DR1455/finance_questions")
        healthcare_dataset = load_dataset("iecjsu/lavita-ChatDoctor-HealthCareMagic-100k")

        keep = ["text", "domain", "label"]

        # Filter and prepare law dataset
        law_data = (
            law_dataset["train"]
            .filter(lambda x: x["prompt"] is not None and x["prompt"].strip() != "")
            .filter(lambda x: all(v is not None for v in x.values()))
            .select(range(min(DATASET_SIZE, len(law_dataset["train"]))))
            .map(
                lambda x: {"text": x["prompt"], "domain": "law", "label": 0},
                remove_columns=[
                    c for c in law_dataset["train"].column_names if c not in keep
                ],
            )
        )

        # Filter and prepare finance dataset
        finance_data = (
            finance_dataset["train"]
            .filter(
                lambda x: x["instruction"] is not None
                and len(str(x["instruction"]).strip()) > 0
            )
            .filter(lambda x: all(v is not None for v in x.values()))
            .select(range(min(DATASET_SIZE, len(finance_dataset["train"]))))
            .map(
                lambda x: {"text": str(x["instruction"]), "domain": "finance", "label": 1},
                remove_columns=[
                    c for c in finance_dataset["train"].column_names if c not in keep
                ],
            )
        )

        # Filter and prepare healthcare dataset
        healthcare_data = (
            healthcare_dataset["train"]
            .filter(lambda x: x["input"] is not None and len(str(x["input"]).strip()) > 0)
            .filter(lambda x: all(v is not None for v in x.values()))
            .select(range(min(DATASET_SIZE, len(healthcare_dataset["train"]))))
            .map(
                lambda x: {"text": str(x["input"]), "domain": "healthcare", "label": 2},
                remove_columns=[
                    c for c in healthcare_dataset["train"].column_names if c not in keep
                ],
            )
        )

        # Concatenate datasets
        combined_dataset = concatenate_datasets([law_data, finance_data, healthcare_data])

        # Split into train and test sets using dataset's train_test_split method
        data = combined_dataset.train_test_split(test_size=TRAIN_SPLIT, seed=SEED)

        test_dataset = data["test"].to_pandas()

        train_dataset = data["train"].to_pandas()

        # shiffle the train dataset
        train_dataset = train_dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)

        train_dataset, eval_dataset = train_test_split(
            train_dataset,
            test_size=TRAIN_SPLIT,
            random_state=SEED,
            stratify=train_dataset["domain"],
        )

        return train_dataset, eval_dataset, test_dataset


    @staticmethod
    def load_ood_test_dataset() -> pd.DataFrame:
        """
        Load the out-of-distribution test dataset.

        Returns:
            pd.DataFrame: DataFrame containing out-of-distribution test examples
        """

        jigsaw_splits = {
            "train": "train_dataset.csv",
            "validation": "val_dataset.csv",
            "test": "test_dataset.csv",
        }
        jigsaw_df = pd.read_csv(
            "hf://datasets/Arsive/toxicity_classification_jigsaw/"
            + jigsaw_splits["validation"]
        )

        jigsaw_df = jigsaw_df[
            (jigsaw_df["toxic"] == 1)
            | (jigsaw_df["severe_toxic"] == 1)
            | (jigsaw_df["obscene"] == 1)
            | (jigsaw_df["threat"] == 1)
            | (jigsaw_df["insult"] == 1)
            | (jigsaw_df["identity_hate"] == 1)
        ]

        jigsaw_df = jigsaw_df.rename(columns={"comment_text": "text"})
        jigsaw_df["label"] = 3
        jigsaw_df["domain"] = "ood"
        jigsaw_df = jigsaw_df[["text", "label", "domain"]]
        jigsaw_df = jigsaw_df.dropna(subset=["text"])
        jigsaw_df = jigsaw_df[jigsaw_df["text"].str.strip() != ""]

        # Load OLID dataset
        olid_splits = {"train": "train.csv", "test": "test.csv"}
        olid_df = pd.read_csv("hf://datasets/christophsonntag/OLID/" + olid_splits["test"])
        olid_df = olid_df.rename(columns={"cleaned_tweet": "text"})
        olid_df["label"] = 3
        olid_df['domain'] = 'ood'
        olid_df = olid_df[["text", "label", "domain"]]
        olid_df = olid_df.dropna(subset=["text"])
        olid_df = olid_df[olid_df["text"].str.strip() != ""]

        # Load hateXplain dataset
        hate_xplain = pd.read_parquet(
            "hf://datasets/nirmalendu01/hateXplain_filtered/data/train-00000-of-00001.parquet"
        )
        hate_xplain = hate_xplain.rename(columns={"test_case": "text"})
        hate_xplain = hate_xplain[(hate_xplain["gold_label"] == "hateful")]
        hate_xplain = hate_xplain[["text", "label"]]
        hate_xplain["label"] = 3
        hate_xplain["domain"] = "ood"
        hate_xplain = hate_xplain.dropna(subset=["text"])
        hate_xplain = hate_xplain[hate_xplain["text"].str.strip() != ""]

        # Load TUKE Slovak dataset
        tuke_sk_splits = {"train": "train.json", "test": "test.json"}
        tuke_sk_df = pd.read_json(
            "hf://datasets/TUKE-KEMT/hate_speech_slovak/" + tuke_sk_splits["test"],
            lines=True,
        )
        tuke_sk_df = tuke_sk_df.rename(columns={"text": "text"})
        tuke_sk_df = tuke_sk_df[tuke_sk_df["label"] == 0]
        tuke_sk_df["label"] = 3
        tuke_sk_df["domain"] = "ood"
        tuke_sk_df = tuke_sk_df[["text", "label", "domain"]]
        tuke_sk_df = tuke_sk_df.dropna(subset=["text"])
        tuke_sk_df = tuke_sk_df[tuke_sk_df["text"].str.strip() != ""]

        try:
            splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
            dkhate = pd.read_parquet("hf://datasets/DDSC/dkhate/" + splits["test"])
            dkhate["label"] = 3
            dkhate["domain"] = "ood"
            dkhate = dkhate.dropna(subset=["text"])
            dkhate = dkhate[dkhate["text"].str.strip() != ""]
        except Exception as e:
            instructions = [
                "Cannot load dkhate dataset. Skipping it.",
                "Error details:"
                "```"
                f"{str(e)}",
                "```",
                "Please check if you are logged in to Hugging Face Hub.",
                "You can do this by running `huggingface-cli login` in your terminal.",
                "Ensure you have access to the dataset: https://huggingface.co/datasets/DDSC/dkhate"
            ]
            for instruction in instructions:
                print(instruction)
            dkhate = pd.DataFrame(columns=["text", "label", "domain"])

        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        web_questions = pd.read_parquet("hf://datasets/Stanford/web_questions/" + splits["test"])

        web_questions['text'] = web_questions['question']
        web_questions['label'] = 3
        web_questions['domain'] = 'ood'
        web_questions['dataset'] = 'web_questions'
        web_questions = web_questions[['text', 'label', 'domain']]

        splits = {'train': 'data/train-00000-of-00001-7ebb9cdef03dd950.parquet', 'test': 'data/test-00000-of-00001-fbd3905b045b12b8.parquet'}
        ml_questions = pd.read_parquet("hf://datasets/mjphayes/machine_learning_questions/" + splits["test"])

        ml_questions['text'] = ml_questions['question']
        ml_questions['label'] = 3
        ml_questions['domain'] = 'ood'
        ml_questions['dataset'] = 'machine_learning_questions'
        ml_questions = ml_questions[['text', 'label', 'domain']]

        ood_datasets = {
            "jigsaw": jigsaw_df,
            "olid": olid_df,
            "hate_xplain": hate_xplain,
            "tuke_sk": tuke_sk_df,
            "dkhate": dkhate,
            "web_questions": web_questions,
            "ml_questions": ml_questions,
        }

        return ood_datasets


def load_train_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function for DataLoader.load_train_dataset()"""
    train_dataset, eval_dataset, _ = DataLoader.load_train_dataset()
    return train_dataset, eval_dataset


def load_dev_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset, eval_dataset, _ = DataLoader.load_train_dataset()
    return train_dataset.sample(DEV_SIZE, random_state=SEED), eval_dataset.sample(DEV_SIZE, random_state=SEED),


def load_id_test_dataset() -> pd.DataFrame:
    """Convenience function for DataLoader.load_id_test_dataset()"""
    _, _, test_dataset = DataLoader.load_train_dataset()
    return test_dataset


def load_ood_test_dataset() -> pd.DataFrame:
    """Convenience function for DataLoader.load_ood_test_dataset()"""
    datasets_dict = DataLoader.load_ood_test_dataset()
    data = []
    for key, dataset in datasets_dict.items():
        dataset["dataset"] = key
        data.append(dataset)
    return pd.concat(data).reset_index(drop=True)

