import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    # Filter the dataset to keep only dictionaries where 'challenge' key is True

    filtered_dataset = dataset.filter(lambda example: example['challenging'] == True)



    return filtered_dataset