# Energy-based model for Domain Generalization.
Domain generalization on chest x-ray data using energy based models. Please see the Proposal document for a detailed overview of the project.

## Running

1. Install the following packages:
```
pip install pytorch torchvision tqdm opencv-python numpy
```

2. Download the PadChest, CheXpert and IU dataset and organize the files by task. The files A, C, N, and P refer to the disease used for classification labeling.

```
├── dataset
│   ├── A
│   │   ├── lateral
│   │   └── pa
│   ├── C
│   │   ├── lateral
│   │   └── pa
│   ├── N
│   │   ├── lateral
│   │   └── pa
│   └── P
│       ├── lateral
│       └── pa
```
        

The file structure allows the dataloader to correctly assign labels for each dataset.

3. Edit the `demo.py` script based on your choice of model, task category, and projection.

4. Run:
```
python demo.py
```
