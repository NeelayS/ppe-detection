import torch
import torch.nn.functional as F


def eval(model, testloader, device):

    n_total = 0
    n_correct = 0

    for data, labels in testloader:

        data = [d.to(device) for d in data]
        labels = torch.LongTensor(labels).to(device)

        preds = model(data)
        preds = [pred[0] for pred in preds]
        preds = torch.cat(preds)
        preds = F.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)

        n_correct += torch.sum(torch.eq(preds, labels))
        n_total += len(labels)

    accuracy = (n_correct / n_total) * 100

    print(f"Accuracy: {accuracy}%")

    return accuracy.item()
