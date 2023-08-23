resolution = 256


from Geometry import *
from PIL import Image
import json

class ImageGenerator:

    """
    Class for random shapes images generation with description.
    
    """

    def __init__(self, export_folder = None):
        self.export_folder = export_folder
        self.n_image = 0

    def generate_n_samples(self, n, save=False):
        """
        Generate n images. If save, they are saved in self.export folder with number as name. And corresponding json.
        """
        self.desc = {} #description dict for training
        self.imgs = []
        for i in range(n):
            img, param_dict= self.generate()
            self.imgs.append(img)
            self.desc[self.n_image] = param_dict
            self.n_image += 1

        if save:
            for i, image in enumerate(self.imgs):
                destination = self.export_folder + '/' + '{:05d}'.format(i)
                image.save(destination + '.png')
                with open(destination + '.json', 'w') as f:
                    json.dump(self.desc[i], f)
        return self.imgs, self.desc
    
            
            
        
    def generate(self):
        shapes = [Circle, Triangle, Hexagon, Rhombus]
        n = np.random.randint(1, 6) # number of shapes
        id = 0
        img_desc = {}
        background_color = np.random.randint(0, 256, 3)
        is_free = np.ones((resolution, resolution)).astype(bool)
        img = np.tile(background_color, (resolution, resolution, 1))
        for i in range(n):
            index = np.random.randint(0, 4)
            figure = shapes[index](is_free = is_free, id = id)
            img, is_free = figure.draw(img, is_free)
            img_desc[id] = figure.get_param_dict()
            id += 1
            
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        return img, img_desc


        