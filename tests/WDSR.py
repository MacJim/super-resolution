from model import resolve_single
from model.wdsr import wdsr_b

from utils import load_image, plot_sample


model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('../weights/wdsr-b-32-x4/weights.h5')

# lr = load_image('../demo/0829x4-crop.png')
lr = load_image('../demo/1.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)