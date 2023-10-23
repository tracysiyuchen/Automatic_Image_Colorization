from util import *
from models import *
from trainer import Trainer
import gc

gc.collect()
gray_path = '../data/gray_scale.npy'
ab_path = '../data/ab/ab1.npy'
images_gray, images_lab = load_data(gray_path, ab_path, num_images=100)
train_data, val_data, test_data = split_data(images_gray, images_lab, [0.7, 0.1, 0.2])
model = MobileNet()
trainer = Trainer(model, batch_size=16)
trainer.train(train_data, val_data, test_data)

