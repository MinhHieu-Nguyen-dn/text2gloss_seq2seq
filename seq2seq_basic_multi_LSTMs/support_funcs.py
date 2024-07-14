from datasets import DatasetDict
import torch
import torch.nn as nn


def strip_bom_and_newline(dataset):
    """
  Strips the BOM character and newline character from all 'gloss' and 'text' fields in the dataset.

  Args:
    dataset: A DatasetDict object.

  Returns:
    A new DatasetDict object with stripped fields.
  """

    def strip_fn(example):
        example['gloss'] = example['gloss'].strip('\ufeff').strip('\n')
        example['text'] = example['text'].strip('\ufeff').strip('\n')
        return example

    return dataset.map(strip_fn)


def split_dataset(dataset, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    """
  Splits a dataset into train, validation, and test sets.

  Args:
    dataset: The dataset to split.
    train_ratio: The ratio of the dataset to use for training.
    validation_ratio: The ratio of the dataset to use for validation.
    test_ratio: The ratio of the dataset to use for testing.

  Returns:
    A DatasetDict containing the train, validation, and test splits.
  """

    total_size = len(dataset)

    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)
    test_size = total_size - train_size - validation_size

    return DatasetDict({
        "train": dataset.select(range(train_size)),
        "validation": dataset.select(range(train_size, train_size + validation_size)),
        "test": dataset.select(range(train_size + validation_size, total_size))
    })


def tokenize_example(example, en_nlp, max_length, lower, sos_token, eos_token):
    gloss_tokens = [token.text for token in en_nlp.tokenizer(example["gloss"])][:max_length]
    text_tokens = [token.text for token in en_nlp.tokenizer(example["text"])][:max_length]
    if lower:
        gloss_tokens = [token.lower() for token in gloss_tokens]
        text_tokens = [token.lower() for token in text_tokens]
    gloss_tokens = [sos_token] + gloss_tokens + [eos_token]
    text_tokens = [sos_token] + text_tokens + [eos_token]
    return {"gloss_tokens": gloss_tokens, "text_tokens": text_tokens}


def numericalize_example(example, gloss_vocab, text_vocab):
    gloss_ids = gloss_vocab.lookup_indices(example["gloss_tokens"])
    text_ids = text_vocab.lookup_indices(example["text_tokens"])
    return {"gloss_ids": gloss_ids, "text_ids": text_ids}


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_gloss_ids = [example["gloss_ids"] for example in batch]
        batch_text_ids = [example["text_ids"] for example in batch]
        batch_gloss_ids = nn.utils.rnn.pad_sequence(batch_gloss_ids, padding_value=pad_index)
        batch_text_ids = nn.utils.rnn.pad_sequence(batch_text_ids, padding_value=pad_index)
        batch = {
            "gloss_ids": batch_gloss_ids,
            "text_ids": batch_text_ids,
        }
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


def translate_sentence(
        sentence,
        model,
        en_nlp,
        # de_nlp,
        gloss_vocab,
        text_vocab,
        lower,
        sos_token,
        eos_token,
        device,
        max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in en_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = text_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = gloss_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == gloss_vocab[eos_token]:
                break
        tokens = gloss_vocab.lookup_tokens(inputs)
    return tokens


def get_tokenizer_fn(nlp, lower):
    def tokenizer_fn(s):
        tokens = [token.text for token in nlp.tokenizer(s)]
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens

    return tokenizer_fn
