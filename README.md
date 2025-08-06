# Summer Internship 2025 VGG16 Sample

## Training

In order to train the VGG16 models with the CIFAR-100 dataset using the framework PyTroch in Python, run the following:

```shell
python -m venv .venv # Create a virtual environment
source .venv/bin/activate  # Activate the virtual environment, not necessary in VS Code
pip install torch torchvision numpy onnx tqdm  # Install the required packages
python main.py
```