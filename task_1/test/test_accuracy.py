import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.curdir))
from task_1.task_1_utils import FullyConnected
from task_1.task_1_config import input_size, output_size, hidden_sizes, DATA_DIR


if __name__ == "__main__":
    training_type = "E2ET"  # "gradual"
    model_path = f"mnist_net_{training_type}_{'_'.join([str(s) for s in hidden_sizes])}.pth"
    model = FullyConnected(input_size, output_size, hidden_sizes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Download the MNIST test set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

    # Evaluate the accuracy on the test set
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images.view(images.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy on the MNIST test set: {:.2%}'.format(accuracy))
