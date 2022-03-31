import torch
import torch.nn.functional as F
from os.path import join
import matplotlib.pyplot as plt


def loss_fn(preds, labels):

    preds = [pred[0] for pred in preds]
    preds = torch.cat(preds)
    # labels = torch.LongTensor(labels)

    loss = F.cross_entropy(preds, labels)

    return loss


def train_classifier(model, dataloader, optimizer, epochs, device, save_dir):

    model = model.to(device)
    model.train()

    iter_losses = []

    for epoch in range(epochs):
        for i, (data, labels) in enumerate(dataloader):

            optimizer.zero_grad()

            data = [d.to(device) for d in data]
            labels = torch.LongTensor(labels).to(device)

            # data, labels = data.to(device), labels.to(device)
            preds = model(data)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f"Iteration {(epoch*len(dataloader)) + i}: Loss = {loss.item()}")
                iter_losses.append(loss.item())

            if (i + 1) % 500 == 0:
                torch.save(
                    model,
                    join(
                        save_dir,
                        "iter" + str((epoch * len(dataloader)) + i) + "_model.pth",
                    ),
                )

    plt.plot(iter_losses)
