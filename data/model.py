import torch
import torch.nn as nn
from dataloader import DHCDataset
import torch.utils.data as data_utils
import torch.optim as optim

device = torch.device("mps")
HIDDEN_SIZE = 512
NUM_CLASSES = 46
learning_rate = 3e-4

dhcd_train = DHCDataset("./DHCD_Dataset/dataset/dataset.npz")
dhcd_test = DHCDataset("./DHCD_Dataset/dataset/dataset.npz", train=False)


train_loader = data_utils.DataLoader(dataset=dhcd_train, batch_size=128, shuffle=True)
test_loader = data_utils.DataLoader(dataset=dhcd_test, batch_size=128, shuffle=False)

convnet = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.Flatten(),
    nn.Linear(64 * 64, 128),
    nn.ReLU(),
    nn.Linear(128, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
)

criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(convnet.parameters(), lr=learning_rate)
convnet = convnet.to(device)
training_losses = []
val_losses = []
num_epochs = 25


@torch.no_grad()
def get_accuracy(dataloader, dataset):
    total_correct = 0
    for epoch in range(num_epochs):
        for i, (X, y_true) in enumerate(dataloader):
            X = X.to(device)
            y_true = y_true.to(device)
            y_preds = convnet(X)
            total_correct += torch.sum(torch.argmax(y_preds, dim=1) == y_true).item()

    return 100 * ((total_correct) / (len(dataset) * num_epochs))


# Training Loop
for epoch in range(num_epochs):
    for i, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        optimiser.zero_grad()
        y_preds = convnet(X_train)
        loss = criterion(y_preds, y_train)
        training_losses.append(loss.item())
        loss.backward()
        optimiser.step()
        if i % 50 == 0:
            print(f"epoch [{epoch+1}/{num_epochs}] : iter [{i}/610] || loss : {loss}")

accuracy = get_accuracy(train_loader, dhcd_train)
print(f"\ntraining accuracy: {accuracy:.2f}%")

# Validation Loop
convnet.eval()
with torch.no_grad():
    for epoch in range(num_epochs):
        for i, (X_val, y_val) in enumerate(test_loader):
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            y_test_preds = convnet(X_val)
            loss = criterion(y_test_preds, y_val)
            val_losses.append(loss.item())
            if (i + 1) % 10 == 0:
                print(
                    f"epoch [{epoch+1}/{num_epochs}] : iter [{i+1}/{len(test_loader)}] || loss : {loss}"
                )

accuracy = get_accuracy(test_loader, dhcd_test)
print(f"\ntest accuracy: {accuracy:.2f}%")

print(f"average training loss: {torch.mean(torch.tensor(training_losses))}")
print(f"average validation loss: {torch.mean(torch.tensor(val_losses))}")

FILE = "dhcd_model.pth"
torch.save(convnet.state_dict(), FILE)
