import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Corruptor')

parser.add_argument('--data_path', type=str, help='data path')
parser.add_argument('--img_size', default=28, type=int, help='size of images')
parser.add_argument('--image_channels', default=1, type=int, help='number of image channels')
parser.add_argument('--corrupt_type', default='triangle', type=str, help='type of corruption you wish to add')
parser.add_argument('--corruptor_size', default=2, type=int, help='height and width of corruption')
parser.add_argument('--corrupt_color', default='bw', type=str, help='color channels for corruption')

args = parser.parse_args()


class Corruptor:
    def __init__(self, data_path, img_size, image_channels):
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
            self.images = self.data.iloc[:, 1:].values.reshape(-1, img_size, img_size, image_channels)
            self.labels = self.data.iloc[:, 0].values
            self.data_path = data_path
            self.img_size = img_size
            self.image_channels = image_channels
            self.data_type = 'csv'

    def get_starting_point(self, corruptor_size):
        x, y = np.random.choice(list(range(corruptor_size, self.img_size - corruptor_size)), size=2)
        return x, y

    def draw_triangle(self, image, starting_point, fill_color, corruptor_size):
        x, y = starting_point
        image = image.astype('float32')
        vertices = np.array([[x, y], [x - corruptor_size, y + corruptor_size], [x + corruptor_size, y + corruptor_size]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=fill_color, thickness=1)
        cv2.fillPoly(image, [pts], color=fill_color)
        return image

    def triangles(self, corruptor_size, corrupt_color):
        if corrupt_color == 'bw':
            fill_color = (255,)
        else:
            fill_color = corrupt_color
        if self.data_type == 'csv':
            for i, image in tqdm(enumerate(self.images), total=self.images.shape[0]):
                x, y = self.get_starting_point(corruptor_size)
                self.images[i] = self.draw_triangle(image, (x, y), fill_color, corruptor_size)
            new_data_path = self.data_path.split('.')[:-1]
            new_data_path[-1] += '_corrupt_triangle.csv'
            columns = ['label'] + [f'pixel{i}' for i in range(self.img_size ** 2)]
            pd.DataFrame(
                np.hstack((self.labels.reshape(-1, 1), self.images.reshape(-1, self.img_size ** 2))),
                columns=columns).to_csv(new_data_path[0], index=False)


if __name__ == '__main__':
    corruptor = Corruptor(args.data_path, args.img_size, args.image_channels)
    if args.corrupt_type == 'triangle':
        corruptor.triangles(args.corruptor_size, args.corrupt_color)
