import torch.utils.data as data

from utils import *
from Inceptiona_v3_model import InceptionV3
from Resnet_models import ResNet18
from EBM_resnet import ResNetEBM18
from train_and_test import train, test, get_maps
from dataloader import ChestXRayDataset
from PIL import ImageFile
from PIL import Image
import torchvision
import cv2

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.multiprocessing.set_start_method('spawn')
    testing = False # val = True or train = False
    iu_test = False # for testing against the IU dataset
    model = 'EBM' # 'resnet' or 'EBM
    ebm = True # flag for training loop
    category = 'P' # A C P
    projection = 'pa' # pa lateral
    # root directories in HPC for our data
    root1 = '../../padchest'
    root2 = "/vast/lp2663/HealthML_data_organized/CheXpert/"
    root3 = "/vast/lp2663/HealthML_data_organized/IU/"
    # model + training detail output
    save_dir = '../../train_results/NEW2_' + model + "_" + category + "_" + projection
    train_transformer, test_transfomer = train_and_test_transformers_for_resnet()
    if not iu_test:
      # We are training and test on half of the data to reduce the time it takes to train. A manual random seed is used for reproducability
        dataset = ChestXRayDataset(root1, root2, category, projection, train_transformer if not testing else test_transfomer, device)
        train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [0.4, 0.1, 0.5], generator=torch.Generator().manual_seed(42))
        dl_train = data.DataLoader(train_dataset, 512, shuffle=True, num_workers=1)
        dl_val = data.DataLoader(val_dataset, 512, shuffle=False, num_workers=1)
    else:
      # This is the dataloader for the IU dataset
        dataset = ChestXRayDataset(root3, None, category, projection, test_transfomer, device)
        dl_val = data.DataLoader(dataset, 512, shuffle=False, num_workers=1)
        dl_train = None
    # Both models declared here
    # resnet18 = ResNet18(num_classes=2)
    resnet18 = ResNetEBM18(num_classes=2) 

    print("Is EBM? ", str(ebm))
    if torch.cuda.is_available():
        resnet18.cuda()
    if testing:
        resnet18.load_state_dict(torch.load(save_dir + '.pth'))
        test(15, resnet18, dl_train, dl_val, "Chest_X_Ray_Dataset", 1e-5, save_dir, device, ebm)
    else:        
        # save_dir = './test.pth'
        model = train(5, resnet18, dl_train, dl_val, "Chest_X_Ray_Dataset", 1e-4, save_dir, device, ebm) # None = dl_test
        torch.save(model.state_dict(), save_dir + '.pth')
