from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import pytorch_lightning as pl


def get_raw_data():
    raw_data = pd.read_csv("./Data/sampled_999986.csv",
                           converters={1: ast.literal_eval})
    raw_data = raw_data[["sents"]]
    # raw_data = raw_data.sample(n=5000, random_state = rand_seed)

    print(raw_data["sents"].apply(lambda x: len(x) - 1).sum())
    raw_data["cumlen"] = raw_data["sents"].apply(
        lambda x: len(x) - 1).cumsum() - 1
    raw_data["len"] = raw_data["sents"].apply(lambda x: len(x) - 1)
    raw_data = raw_data.set_index("cumlen")

    return raw_data


class AmazonDataset(Dataset):
    def __init__(self, data, raw_data, tokenizer, sent_length):
        self.data = data
        self.len = raw_data["sents"].apply(lambda x: len(x) - 1).sum()
        self.tokenizer = tokenizer
        self.sent_length = sent_length
        self.raw_data = raw_data

    def __len__(self):
        return self.len

    def to_token(self, sentence):
        return self.tokenizer.encode(sentence, max_length=self.sent_length,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")[0]

    def get_pair(self, idx):
        iidx = idx
        while iidx not in self.raw_data.index:
            iidx += 1
        line = self.raw_data["sents"].loc[iidx]
        base = idx - iidx - 2
        return (line[base], line[base + 1])

    def __getitem__(self, index):
        context, input = self.get_pair(index)
        return self.to_token(context), self.to_token(input)


class AmazonDataModule(pl.LightningDataModule):
    def __init__(self, raw_data, batch_size, tokenizer, sent_length):
        super().__init__()
        train_dataset, val_dataset = train_test_split(raw_data, test_size=0.01)
        self.train = AmazonDataset(
            train_dataset, raw_data, tokenizer, sent_length)
        self.test = AmazonDataset(
            val_dataset, raw_data, tokenizer, sent_length)
        self.val = AmazonDataset(val_dataset, raw_data, tokenizer, sent_length)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4)
