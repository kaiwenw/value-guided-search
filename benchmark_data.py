import datasets


def get_dataset(name):
    # return a dataset with columns "problem" and "answer"
    if name == "aime-24":
        dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
        dataset = dataset.rename_column("Problem", "problem").rename_column("Answer", "answer")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "aime-25":
        dataset = datasets.load_dataset("yentinglin/aime_2025", split="train")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "hmmt-feb-24":
        dataset = datasets.load_dataset("MathArena/hmmt_feb_2024", split="train")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "hmmt-feb-25":
        dataset = datasets.load_dataset("MathArena/hmmt_feb_2025", split="train")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
