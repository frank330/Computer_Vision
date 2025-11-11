import os
import torch
import torch.nn as nn
from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model_weight_path = "./vgg16-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1

    net = vgg(model_name="vgg16")
    net.load_state_dict(torch.load(model_weight_path, map_location=device))




if __name__ == '__main__':
    main()
