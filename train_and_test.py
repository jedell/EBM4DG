import torch
import torch.nn as nn
import tqdm
import datetime
from utils import *


def train(epochs, model, train_dataloader, val_dataloader, dataset_name, lr, save_dir, device, ebm=False):
    print("Starting training ...")
    print("Training on " + dataset_name + " dataset")
    print("LENGTH OF TRAIN DATASET: ", len(train_dataloader.dataset))

    loss_fn = nn.CrossEntropyLoss()
    dom_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm.tqdm(range(epochs))
    for i in pbar:
        print('='*20)
        print(f'Starting epoch {i + 1}/{epochs}')
        print('='*20)

        train_iter_loss = 0.0
        dom_iter_loss = 0.0
        test_iter_loss = 0.0
        train_num_correct = 0

        model.train()
        samples_processed = 0

        for train_step, (image, label, dom_label) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
                dom_label = dom_label.cuda()
            optimizer.zero_grad()

            if ebm:
                pred, dom_pred = model(image)
                # print(pred, dom_pred)
                class_loss = loss_fn(pred, label)
                dom_loss = dom_loss_fn(dom_pred, dom_label)
                neg_dom_loss = -dom_loss
                neg_dom_loss.backward(retain_graph=True)
                class_loss.backward()
                optimizer.step()
                train_iter_loss += class_loss.item()
                dom_iter_loss += dom_loss.item()

                ebm_class_predictions = torch.argmax(pred, dim = 1).float()
                # print(label)
                num_correct = torch.eq(label, ebm_class_predictions).sum().item()
                train_num_correct += num_correct

            else:
                pred = model(image)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                train_iter_loss += loss.item()

                class_preds = torch.argmax(pred, dim = 1).float()
                num_correct = torch.eq(label, class_preds).sum().item()
                train_num_correct += num_correct
                
        train_iter_loss /= (train_step + 1)
        print(train_num_correct, len(train_dataloader.dataset))
        train_acc = train_num_correct / len(train_dataloader.dataset)
        print(f'Training Loss: {train_iter_loss:.4f}')
        print(f'Training Acc: {train_acc:.4f}')
        if ebm:
            dom_iter_loss /= (train_step + 1)
            print(f'Domain Classifier Loss: {dom_iter_loss:.4f}')

        with open(save_dir + '.txt', "a") as f:
            f.write(str(train_iter_loss) + "," + str(train_acc) + "," + str(dom_iter_loss if ebm else "-1") + "\n")
    print('Training complete..')
    print("Evaluating at step, ", train_step)
    print("Testing EBM?" + str(ebm))
    correct = 0
    total = 0
    test_iter_loss = 0.0
    accuracy = 0
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for val_step, (image, label, _) in enumerate(val_dataloader):
            label = label.to(device)
            if ebm:
                pred, _ = model(image)
                class_loss = loss_fn(pred, label)
                # dom_loss = dom_loss_fn(dom_pred, dom_label)
                # neg_dom_loss = -dom_loss
                # neg_dom_loss.backward(retain_graph=True)
                test_iter_loss += class_loss.item()
                # dom_iter_loss += dom_loss.item()
                ebm_class_predictions = torch.argmax(pred, dim = 1).float()
                total += label.size(0)
                correct += torch.eq(label, ebm_class_predictions).sum().item()
                print("NUM CORRECT ebm:", correct, "/", total)
                # accuracy = 100 * correct / total
            else:
                pred = model(image)
                loss = loss_fn(pred, label)
                test_iter_loss += loss.item()
                # _, preds = torch.max(pred, 1)
                preds = torch.argmax(pred, dim = 1).float()
                total += label.size(0)
                correct += torch.eq(label, preds).sum().item()
                print("NUM CORRECT resnet:", correct, "/", total)

    print("Test Loss:", test_iter_loss / total, "\nTest Acc:", (correct * 100) / total, "\nTotal Samples:", total, len(val_dataloader.dataset))
    return model

def get_maps(model, val_dataloader, device, ebm, save_dir):
    count = 0
    model.eval()
    for val_step, (image, label, _) in enumerate(val_dataloader):
        label = label.to(device)
        if ebm:
            pred, _ = model(image)
            saliency(image, pred, count, save_dir)
            # dom_loss = dom_loss_fn(dom_pred, dom_label)
            # neg_dom_loss = -dom_loss
            # neg_dom_loss.backward(retain_graph=True)
            # dom_iter_loss += dom_loss.item()
            # accuracy = 100 * correct / total
        else:
            pred = model(image)
            saliency(image, pred, count, save_dir)
        count += 1
        if count > 4:
            return

def test(epochs, model, train_dataloader, val_dataloader, dataset_name, lr, save_dir, device, ebm):
    print("Testing EBM?" + str(ebm))
    print(datetime.datetime.now().isoformat())
    print(val_dataloader.dataset.root_dir_1, val_dataloader.dataset.root_dir_2, val_dataloader.dataset.category, val_dataloader.dataset.projection)
    correct = 0
    total = 0
    test_iter_loss = 0.0
    accuracy = 0
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.eval()
    for val_step, (image, label, _) in enumerate(val_dataloader):
        label = label.to(device)
        if ebm:
            pred, _ = model(image)
            class_loss = loss_fn(pred, label)
            # dom_loss = dom_loss_fn(dom_pred, dom_label)
            # neg_dom_loss = -dom_loss
            # neg_dom_loss.backward(retain_graph=True)
            test_iter_loss += class_loss.item()
            # dom_iter_loss += dom_loss.item()
            ebm_class_predictions = torch.argmax(pred, dim = 1).float()
            print(label, ebm_class_predictions)
            total += label.size(0)
            correct += torch.eq(label, ebm_class_predictions).sum().item()
            print("NUM CORRECT:", correct, "/", total)
            # accuracy = 100 * correct / total
        else:
            pred = model(image)
            loss = loss_fn(pred, label)
            test_iter_loss += loss.item()
            # _, preds = torch.max(pred, 1)
            _, preds = torch.max(pred, dim = 1)
            print("OUT OF MAX", _, pred)
            total += label.size(0)
            print(label, preds)
            correct += torch.eq(label, preds).sum().item()
            print("NUM CORRECT resnet:", correct, "/", total)

    print("Test Loss:", test_iter_loss / total, "\nTest Acc:", (correct * 100) / total, "\nTotal Samples:", total, len(val_dataloader.dataset))
    return 

