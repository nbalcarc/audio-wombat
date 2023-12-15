from birds import audio_wombat
import numpy as np
from numpy import typing as npt
import csv


# Resolution of each image: 64x491 (stride of 64)
STRIDE = 64


class LabelGroup():
    """One per species of bird."""
    def __init__(self, label: str, frames: npt.NDArray[np.uint8], learning_rate: float): # number of frames is (number of birds, 1000)
        self.label = label #the species
        self.frames = frames #(1000, )
        self.frames_pointer = self.frames.ctypes.data
        self.weights = np.ones(frames.size)
        self.bias = 0
        self.learning_rate = learning_rate
        
    def train(self, labels: npt.NDArray[np.uint8], images: npt.NDArray[np.uint8]):
        """images: [[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],[0,0,0,0,0,,1,1,1,1,1,101,01,01,00,0,0k,,0,0,0,0,0,], [0,0,0,0,0,0,0,0,]]"""
        # images: (250 * 1000, x)
        # images: list of spectrograms, where each spectrogram corresponds to a single training example
        # and the label at the same index corresponds to the label for that spectrogram
        output_buffer = np.zeros(self.frames.size, dtype = np.uint32) #the scores for each frame will be stored here
        output_pointer = output_buffer.ctypes.data #creates a pointer to the numpy array
        for image in images:
            image_pointer = image.ctypes.data #creates a pointer to the numpy array
            audio_wombat(self.frames_pointer, image_pointer, output_pointer, image.size, STRIDE, self.frames.size) #calculate the confidences
            old_weights = self.weights.copy()
            predictions = (self.weights * (-1 * output_buffer) + self.bias)
            #num_accurate_predictions = ((predictions >= 0) == (labels == 1)).sum()
            self.weights = self.learning_rate * predictions * labels #change weights according to confidence
            self.bias += 1 if (sum(self.weights) - sum(old_weights) >= 0) else -1

    def predict(self, image: npt.NDArray[np.uint8]):
        """Predict one image for this bird."""
        frames_pointer = self.frames.ctypes.data
        image_pointer = image.ctypes.data
        output_buffer = np.zeros(self.frames.size, dtype = np.uint32)
        output_pointer = output_buffer.ctypes.data #creates a pointer to the numpy array
        audio_wombat(frames_pointer, image_pointer, output_pointer, image.size, STRIDE, self.frames.size)
        return max(self.weights * (-1 * output_buffer) + self.bias)


class Wombat():
    def __init(self, labels: list[str], frames: npt.NDArray[np.uint8], learning_rate: float):
        self.learning_rate = learning_rate
        self.label_groups: list[LabelGroup] = []

        bird_data: dict[str, list[npt.NDArray[np.uint8]]] = {label:[] for label in set(labels)}
        for label, frame in zip(labels, frames):
            bird_data[label].append(frame)

        for label in set(labels):
            self.label_groups.append(LabelGroup(label, bird_data[label], learning_rate))


    def train(self, labels: list[str], frames: npt.NDArray[np.uint8]):
        bird_data: dict[str, list[npt.NDArray[np.uint8]]] = {label:[] for label in set(labels)}
        for label, frame in zip(labels, frames):
            bird_data[label].append(frame)

        for label_group in self.label_groups:
            label_group.train(LabelGroup([1 if label_group.label==label else -1 for label in labels], bird_data, self.learning_rate))

    def predict(self, image):
        return [label_group.label for label_group in self.label_groups if label_group.predict(image)]
            


def load_file(filepath):
    file_path = filepath

    label_list = []
    data_list = []

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not label_list:
                label_list.append(row[0])
            else:
                data_list.extend(row[1:])

    data_array = np.array(data_list, dtype=np.uint8)

    return label_list, data_array




