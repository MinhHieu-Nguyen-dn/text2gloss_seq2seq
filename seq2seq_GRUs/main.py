import torch.optim as optim
import numpy as np
import spacy
import datasets
import torchtext
from torchtext import vocab
import tqdm
import evaluate

from support_funcs import *
from models import *

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

dataset_ASL = datasets.load_dataset("aslg_pc12")
dataset_ASL = strip_bom_and_newline(dataset_ASL)

dataset = split_dataset(dataset_ASL["train"])
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

en_nlp = spacy.load("en_core_web_sm")

max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "en_nlp": en_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

gloss_vocab = vocab.build_vocab_from_iterator(
    train_data["gloss_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

text_vocab = vocab.build_vocab_from_iterator(
    train_data["text_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

assert gloss_vocab[unk_token] == text_vocab[unk_token]
assert gloss_vocab[pad_token] == text_vocab[pad_token]

unk_index = gloss_vocab[unk_token]
pad_index = gloss_vocab[pad_token]

gloss_vocab.set_default_index(unk_index)
text_vocab.set_default_index(unk_index)

fn_kwargs = {"gloss_vocab": gloss_vocab, "text_vocab": text_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

data_type = "torch"
format_columns = ["gloss_ids", "text_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

batch_size = int(input('Batch size? = '))

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

input_dim = len(text_vocab)
output_dim = len(gloss_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training using: {}'.format(device))

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)
model.apply(init_weights)
print(model)
print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")
model_name = "text2gloss-aslg_pc12-model-{}.pt".format('01')

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_name)
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

model.load_state_dict(torch.load(model_name))

test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")


translations = [
    translate_sentence(
        example["text"],
        model,
        en_nlp,
        gloss_vocab,
        text_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )
    for example in tqdm.tqdm(test_data)
]

bleu = evaluate.load("bleu")

predictions = [" ".join(translation[1:-1]) for translation in translations]

references = [[example["gloss"]] for example in test_data]

tokenizer_fn = get_tokenizer_fn(en_nlp, lower)

results = bleu.compute(
    predictions=predictions, references=references, tokenizer=tokenizer_fn
)

print(results)
