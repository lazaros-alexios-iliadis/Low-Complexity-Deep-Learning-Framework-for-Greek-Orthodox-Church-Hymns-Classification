import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import os
import numpy as np
import random
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
from VGG_models import MicroVGG

# Torch version
print('Torch version: ', torch.__version__)

#  Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


random_seed = 42
seed_everything(random_seed)

cudnn.benchmark = True

data_path = "Mel"


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


walk_through_dir(data_path)


train_dir = "D:\\Projects\\Ymnodos\\AppliedSciences\\Data\\train\\"
val_dir = "D:\\Projects\\Ymnodos\\AppliedSciences\\Data\\val\\"
test_dir = "D:\\Projects\\Ymnodos\\AppliedSciences\\Data\\test\\"

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4680, 0.4225, 0.4795], std=[0.4680, 0.4225, 0.4795]),
])

train_data = datasets.ImageFolder(train_dir, data_transforms)
val_data = datasets.ImageFolder(val_dir, data_transforms)
test_data = datasets.ImageFolder(test_dir, data_transforms)

print(f"Train data:\n{train_data}\n Validation data:\n{val_data}\nTest data:\n{test_data}")

class_names = train_data.classes
print(class_names)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

model = MicroVGG(num_classes=23).to(device)
print(model)

start = timer()
summary(model, input_size=[1, 3, 64, 64])  # do a test pass through of an example input size
end = timer()
print(f"Total training time: {end - start:.3f} seconds")


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):

    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }
    best_accuracy = 0
    j = 0

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if test_acc > best_accuracy:
            torch.save(model.state_dict(), 'best-model-parameters.pt')
            best_accuracy = test_acc
            j = epoch
    # 6. Plot results and print the best achieved accuracy

    plt.plot(range(epochs), results["train_loss"], label='Train')
    plt.plot(range(epochs), results["test_loss"], label='Validation')
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.legend(fontsize=24)
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.savefig('loss.png', dpi=300)
    plt.show()

    plt.plot(range(epochs), results["train_acc"], label='Train')
    plt.plot(range(epochs), results["test_acc"], label='Validation')
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.legend(fontsize=24)
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.savefig('accuracy.png', dpi=300)
    plt.show()

    print("Best accuracy: ", best_accuracy * 100, "%")
    print("Best performance in epoch: ", j)

    return results


torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 100

# Recreate an instance of the model
instance = MicroVGG(num_classes=23).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=instance.parameters(), lr=0.0003)

# Start the timer
start_time = timer()

# Train model_0
model_results = train(model=instance,
                      train_dataloader=train_loader,
                      test_dataloader=test_loader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")
