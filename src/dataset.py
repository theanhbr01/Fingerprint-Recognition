import os
from PIL import Image
import numpy as np


class DataSet:
    def __init__(self, dataset_path = '../data/fvc2000', prefix_label = 'fvc2000', image_height = 192, image_width = 192, logging = False) -> None:
        self.dataset_path = dataset_path
        self.image_height = image_height
        self.image_width = image_width

        self.images = [] 
        self.labels = []

        for filename in os.listdir(dataset_path):
            image_path = os.path.join(dataset_path, filename)
            image = Image.open(image_path)

            label, index = filename.split('_')
            label = int(label)
            
            self.images.append(np.array(image))
            self.labels.append(label)

            if(logging):
                print('> loaded %s with label %s' % (filename, label) )

        if(logging):
                print('> imported %s images' % (self.images.length) )

    def File_path(self, file_name):
        return os.path.join(self.dataset_path, file_name)