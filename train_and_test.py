import torch
import torch.nn as nn
import tqdm


def train(epochs, model, train_dataloader, val_dataloader, dataset_name, lr):
    print("Starting training ...")
    print("Training on " + dataset_name + " dataset")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(epochs))
    for i in pbar:
        print('='*20)
        print(f'Starting epoch {i + 1}/{epochs}')
        print('='*20)

        train_iter_loss = 0.0
        test_iter_loss = 0.0

        model.train()

        for train_step, (image, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_iter_loss += loss.item()

            if train_step % 20 == 0:
                print("Evaluating at step, ", train_step)
                accuracy = 0
                model.eval()
                for val_step, (image, label) in enumerate(val_dataloader):
                    pred = model(image)
                    loss = loss_fn(pred, label)
                    test_iter_loss += loss.item()
                    _, preds = torch.max(pred, 1)
                    accuracy += sum(preds == label).numpy()
                test_iter_loss /= (val_step + 1)
                accuracy = accuracy/len(val_dataloader)
                print(f'Validation Loss: {test_iter_loss:.4f}, Accuracy: {accuracy:.4f}')
                model.train()
        train_iter_loss /= (train_step + 1)
        print(f'Training Loss: {train_iter_loss:.4f}')
    print('Training complete..')

