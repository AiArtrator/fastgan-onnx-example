import torch
from lightweight_gan import *

def cast_list(el):
    return el if isinstance(el, list) else [el]

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# Load the model
def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 256,
    optimizer = 'adam',
    fmap_max = 512,
    transparent = False,
    greyscale = False,
    batch_size = 10,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    generate_types = ['default', 'ema'],
    generate_interpolation = False,
    aug_test = False,
    aug_prob=None,
    aug_types=['cutout', 'translation'],
    dataset_aug_prob=0.,
    attn_res_layers = [32],
    freq_chan_attn = False,
    disc_output_size = 1,
    dual_contrast_loss = False,
    antialias = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = None,
    num_workers = None,
    multi_gpus = False,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    seed = 42,
    amp = False,
    show_progress = False,
    use_aim = False,
    aim_repo = None,
    aim_run_hash = None,
    load_strict = True
):
    num_image_tiles = default(num_image_tiles, 4 if image_size > 512 else 8)

    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        attn_res_layers = cast_list(attn_res_layers),
        freq_chan_attn = freq_chan_attn,
        disc_output_size = disc_output_size,
        dual_contrast_loss = dual_contrast_loss,
        antialias = antialias,
        image_size = image_size,
        num_image_tiles = num_image_tiles,
        optimizer = optimizer,
        num_workers = num_workers,
        fmap_max = fmap_max,
        transparent = transparent,
        greyscale = greyscale,
        lr = learning_rate,
        save_every = save_every,
        evaluate_every = evaluate_every,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        amp = amp,
        load_strict = load_strict
    )

    if generate:
        model = Trainer(**model_args, use_aim = use_aim)
        model.load(load_from)

        return model

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


@torch.no_grad()
def generate_(self, G, style, num_image_tiles = 8):
    generated_images = evaluate_in_chunks(8, G, style)
    return generated_images.clamp_(0., 1.)


_model = train_from_folder(generate = True, load_from = 1)

_model.GAN.GE.to('cpu')

imgs = lambda X : _model.generate_(_model.GAN.GE, X)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


model = Wrapper(imgs)

#print(imgs)

# Export the model
torch.onnx.export(model, torch.randn(1, 256), 'model_.onnx', verbose=True,
    export_params=True,
    opset_version=12,
    input_names=["x"],)

import onnxruntime as ort

sess = ort.InferenceSession('model_.onnx')
import numpy as np
x = sess.run(None,{"x": np.random.randn(1, 256).astype(np.float32)})

print(x)