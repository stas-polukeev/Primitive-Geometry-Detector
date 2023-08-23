from typing import Any
import torch 
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from ImageGenerator import ImageGenerator

#it will do inplace generation from image generation, reading from files is not implemented (for now)
class ShapesDataset(Dataset):

    def __init__(self, size, batch_size=64) -> None:
        super().__init__()
        self.size = size
        self.batch_size = batch_size
        self.generator = ImageGenerator()
        self.one_hot = F.one_hot(torch.arange(4))
        self.names = ['Circle', 'Hexagon', 'Rhombus', 'Triangle']
        self.name_encode = {name : i for i, name in enumerate(self.names)}
        self.name_decode = {i : name for i, name in enumerate(self.names)}
        self.figure_keys = ['name', 'y', 'x', 'h', 'w']

    def __len__(self):
        return self.size
    
    def __getitem__(self, index: Any) -> Any:
        
        img_batch, res_batch = [], []
        for i in range(self.batch_size):
            img, dic = self.generator.generate()
            res = torch.zeros((1, 9, 8, 8))
            
            for key in dic.keys():
                name, y, x, h, w = [dic[key][i] for i in self.figure_keys]
                center = torch.tensor([y + h // 2, x + w // 2])
                position = center // 32
                dy, dx = (center - position * 32) / 32
                dh, dw = h / 256, w / 256
                class_num = self.name_encode[name]
                #confidence, bbox_params, class_probas
                res_vec = torch.concat([torch.tensor([1, dy, dx, dh, dw]), self.one_hot[class_num]])
                res[0, :, position[0], position[1]] = res_vec
            img_batch.append(ToTensor()(img).unsqueeze(0))
            res_batch.append(res)
        
        return torch.concat(img_batch, 0), torch.concat(res_batch, 0)