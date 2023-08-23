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
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.images_seen = 0
        self.size = size
        self.batch_size = batch_size
        self.generator = ImageGenerator()
        self.one_hot = F.one_hot(torch.arange(4))
        self.names = ['Circle', 'Hexagon', 'Rhombus', 'Triangle']
        self.imgs_per_figure = {i: 0 for i in self.names}
        self.name_encode = {name : i for i, name in enumerate(self.names)}
        self.name_decode = {i : name for i, name in enumerate(self.names)}
        self.figure_keys = ['name', 'y', 'x', 'h', 'w']

    def __len__(self):
        return self.size
    
    def __getitem__(self, index: Any) -> Any:
        """
        technically gives you batch of size self.batch_size and does not depend on index.
        """
        img_batch, res_batch = [], []
        for i in range(self.batch_size):
            img, dic = self.generator.generate()
            res = torch.zeros((1, 9, 8, 8))
            seen = set()
            for key in dic.keys():
                name, y, x, h, w = [dic[key][i] for i in self.figure_keys]
                if name not in seen:
                    self.imgs_per_figure[name] += 1
                    seen.add(name)
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
            self.images_seen += 1
        
        return torch.concat(img_batch, 0).to(self.device), torch.concat(res_batch, 0).to(self.device)