import torch
from model import ShapesDetector
from loss import yolo_loss
from dataset import ShapesDataset

class Trainer:
    """
    Class for training organization and logging
    """

    def __init__(self, model, train_size, test_size, ) -> None:
        