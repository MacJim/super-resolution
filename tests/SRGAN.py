from model import resolve_single
from model.srgan import generator

from utils import load_image, plot_sample


model = generator()
model.load_weights('../weights/srgan/gan_generator.h5')

lr = load_image('../demo/0869x4-crop.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)