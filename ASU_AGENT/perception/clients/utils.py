import numpy as np



class FrameHandler(object):
    def __init__(self):
        self.img = None
        self.id = 0

    def preprocess(self, frame, shape):
        img = np.array(frame.raw_data)
        img = img.reshape(shape)
        self.img = img[:, :, :3]
        print(f"frame-id: {self.id}\t frame-shape: {self.img.shape}\t")
        self.id += 1
        return self.img


def clean_world(actors_list):
    print("Attempting to clean all actors...!")
    for actor in actors_list:
        actor.destroy()
    print("-"*11, "Cleaned All!", "-"*11)