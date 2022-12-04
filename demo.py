import torch.utils.data as data

from ml4health.utils import *
from ml4health.Inceptiona_v3_model import InceptionV3
from ml4health.Resnet_models import ResNet18
from train_and_test import train


def load_data(train_p_address, train_c_address, train_a_address, train_n_address,
                        test_p_address, test_c_address, test_a_address, test_n_address,
                        model_name, batch_size=6):
    if model_name == 'ResnNet18':
        train_transformer, test_transfomer = train_and_test_transformers_for_resnet()
    elif model_name == 'InceptionV3':
        train_transformer, test_transfomer = train_and_test_transformers_for_inceptionav3()

    train_dir = {
        'P': train_p_address,
        'C': train_c_address,
        'A': train_a_address,
        'N': train_n_address
    }
    train_dataset = ChestXRayDataset(train_dir, train_transformer)

    test_dir = {
        'P': test_p_address,
        'C': test_c_address,
        'A': test_a_address,
        'N': test_n_address
    }
    test_dataset = ChestXRayDataset(test_dir, test_transfomer)

    # Dataloader
    dl_train = data.DataLoader(train_dataset, batch_size, shuffle=True)
    dl_test = data.DataLoader(test_dataset, batch_size, shuffle=True)

    return dl_train, dl_test


if __name__ == '__main__':
    dl_train, dl_test = load_data()
    resnet18 = ResNet18()
    inception = InceptionV3()
    save_dir = ''
    model = train(20, resnet18, dl_train, dl_test, "Chest_X_Ray_Dataset", 1e-5, save_dir)
