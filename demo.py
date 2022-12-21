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
    testing = False # val or train
    iu_test = False
    model = 'EBM' # 'resnet'
    ebm = True
    category = 'P' # A C P
    projection = 'pa' # pa lateral
    root1 = '../../padchest'
    root2 = "/vast/lp2663/HealthML_data_organized/CheXpert/"
    root3 = "/vast/lp2663/HealthML_data_organized/IU/"
    save_dir = '../../train_results/NEW2_' + model + "_" + category + "_" + projection
    print(save_dir)
    # dl_train, dl_test = load_data()
    train_transformer, test_transfomer = train_and_test_transformers_for_resnet()
    if not iu_test:
        dataset = ChestXRayDataset(root1, root2, category, projection, train_transformer if not testing else test_transfomer, device)
        # num_train = round((round(len(dataset) * 0.4)) * 0.8)
        # num_val = (round(len(dataset) * 0.4)) - num_train
        # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
        # sequential split, can be replicated now for validation, make sure range is the same to replicate
        # note, we trained on num_train + num_val by accident (oops), thus the val data loader is using a subset past this for all testing
        # train_dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), 2))
        # val_dataset = torch.utils.data.Subset(dataset, range(num_train + num_val, num_train + num_val + num_val))
        train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [0.4, 0.1, 0.5], generator=torch.Generator().manual_seed(42))
        dl_train = data.DataLoader(train_dataset, 512, shuffle=True, num_workers=1)
        dl_val = data.DataLoader(val_dataset, 512, shuffle=False, num_workers=1)
    else:
        dataset = ChestXRayDataset(root3, None, category, projection, test_transfomer, device)
        dl_val = data.DataLoader(dataset, 512, shuffle=False, num_workers=1)
        dl_train = None
    # 80 / 20,                         #0.2 is the % of data we are using
    # CHANGE MODELS
    # resnet18 = ResNet18(num_classes=2)
    resnet18 = ResNetEBM18(num_classes=2) 

    print("Is EBM? ", str(ebm))
    if torch.cuda.is_available():
        resnet18.cuda()
    if testing:
        resnet18.load_state_dict(torch.load(save_dir + '.pth'))
        get_maps(resnet18, dl_val, device, ebm, save_dir)
        # test(15, resnet18, dl_train, dl_val, "Chest_X_Ray_Dataset", 1e-5, save_dir, device, ebm)
    else:        
        # save_dir = './test.pth'
        model = train(5, resnet18, dl_train, dl_val, "Chest_X_Ray_Dataset", 1e-4, save_dir, device, ebm) # None = dl_test
        torch.save(model.state_dict(), save_dir + '.pth')



    exit()
    # resnet18.load_state_dict(torch.load(save_dir + '.pth'))

    # file_path = "/vast/lp2663/HealthML_data_organized/CheXpert/C/pa/108743v.jpg"
    file_path = "/vast/lp2663/HealthML_data_organized/IU/C/pa/1897.png"
    # file_path = "../../padchest/C/pa/127522431331980806308_00-067-166.png"

    image = image.reshape(1, 3, 224, 224)
    image = image.to(device)
    image.requires_grad_()

    output = resnet18(image)

# Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    image = image.reshape(3, 224, 224)
    image = inv_normalize(image)
    print(image)
    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.savefig("../../train_results/resnet_C_PC_map.png")

    exit()