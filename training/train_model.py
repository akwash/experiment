import torch


def train(model, dataloader, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for points, features, labels in dataloader:

        points = points.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(points, features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        print("loss:", loss.item())