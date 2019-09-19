from devmood.nn.custom.tadziu_dataset import TadziuDataset
from devmood.nn.models.word2vec import SkipGram

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def learn(args):
    dataset = TadziuDataset(args)
    print(f"amount of data: {len(dataset)}")
    batches = DataLoader(dataset=dataset, batch_size=args["batch_size"], shuffle=True, num_workers=6)

    vocab_size = dataset.vocab_size
    FAKE = dataset.FAKE

    criterion = nn.NLLLoss(ignore_index=FAKE)

    model = SkipGram(vocab_size, args["dims"], FAKE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["eta"])

    num_batches = len(dataset) // args["batch_size"]
    print(f"number of upcoming batches: {num_batches}")

    for epoch in range(args["epochs"]):
        tick = time.time()
        loss_total = 0.0
        batch_count = 0
        for target, context in batches:
            batch_count += 1
            loss = criterion(model(context), target)
            loss_total += loss.item()

            if not batch_count % 10 and batch_count > 1:
                print(f"loss of batch {batch_count} during epoch {epoch + 1} is: {loss.item() / batch_count}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("~~~~~~~~~~~~~~~~~~")
        print(f"end of epoch: {epoch + 1} out of {args['epochs']}")
        print(f"time per epoch: {time.time() - tick}")
        print(f"mean loss: {loss_total / num_batches}")
        print(f"total loss: {loss_total}")
        print("~~~~~~~~~~~~~~~~~~")

        weights_in = model.embedding_layer_in.weight.detach().numpy()
        # weights_out = model.embedding_layer_out.weight.detach().numpy()
        # weights = (weights_in + weights_out) / 2
        words = dataset.vocab

    save_weights(dataset, weights_in, words, epoch + 1)
    save_results(args, model.__class__.__name__, loss_total, loss_total / num_batches)


def save_weights(dataset, weights, words, epoch):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              f"../results/weights/{datetime.now()}.csv"), "w") as fout:
        for word in words:
            vec = weights[dataset.word2idx[word]].reshape(1, -1)[0]
            fout.write(f"{word},{' '.join(vec.astype(str).tolist())}\n")

def save_results(args, model_name, loss, mean_loss):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              f"../results/{model_name}-results.json"), "a") as fout:
        fout.write(json.dumps({
            "window": args["window"],
            "batch": args["batch_size"],
            "dims": args["dims"],
            "eta": args["eta"],
            "epochs": args["epochs"],
            "file": os.path.basename(args["input"]),
            "total loss": loss,
            "mean loss": mean_loss
        }, indent=2) + ",\n")

