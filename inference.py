import argparse, os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from main import instantiate_from_config
import random
import cv2

def crop_to_size(img, crop):
    h, w, = img.shape[:2]
    img = img[(h - crop) // 2:(h + crop) // 2,
         (w - crop) // 2:(w + crop) // 2]
    return img

def format_process(img):
    img = (img / 127.5 - 1.0).astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    return torch.from_numpy(img)

def get_img(img_path, device=None):
    img = Image.open(img_path)
    img = np.array(img).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
        
    min_size = min(img.shape[0], img.shape[1])
    img = crop_to_size(img, min_size)
    img = cv2.resize(img, (256, 256))
    img_t = format_process(img)
    img_t = img_t.to(device=device)
    return img_t

def make_batch(rgb_path, infrared_path, device):

    return {
            "rgb": get_img(rgb_path, device),
            "infrared": get_img(infrared_path, device)
        }

def post_process(img):
    img = img.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) / 2.0)[0]
    img = img.permute(1, 2, 0).numpy() * 255 
    return img.astype(np.uint8)

class Inferen(nn.Module):
    def __init__(self, model, steps):
        super(Inferen, self).__init__()
        self.model = model
        self.steps = steps

    def forward(self, batch):
        c1 = self.model.get_learned_conditioning(batch["rgb"])
        c2 = self.model.get_learned_conditioning(batch["infrared"])
        samples_ddim, _ = self.model.sample_log(cond1=c1, cond2=c2, batch_size=1, ddim=True,
                                           ddim_steps=self.steps, eta=1.0)

        return samples_ddim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb_path",
        default="./instance/rgb",
        type=str,
        nargs="?",
        help="dir containing rgb images",
    )
    parser.add_argument(
        "--ir_path",
        default="./instance/ir",
        type=str,
        nargs="?",
        help="dir containing infrared images",
    )
    parser.add_argument(
        "--outdir",
        default='./instance/output',
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--config",
        default='./informationSynthesis/Image2Image.yaml',
        type=str,
        nargs="?",
        help="dir for config",
    )
    parser.add_argument(
        "--checkpoint",
        default='./checkpoints/last.ckpt',
        type=str,
        nargs="?",
        help="dir for config",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="random seed",
    )
    opt = parser.parse_args()

    rgb_path = opt.rgb_path
    infrare_path = opt.ir_path
    rgbs = [rgb_path + '/' + os.path.basename(f) for f in os.listdir(rgb_path)]
    infrareds = [infrare_path + '/' + os.path.basename(f) for f in os.listdir(infrare_path)]

    print(f"Found {len(rgbs)} inputs.")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.checkpoint, map_location='cuda:0')["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    inference_model = Inferen(model, opt.steps)

    os.makedirs(opt.outdir, exist_ok=True)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    with torch.no_grad():
        with model.ema_scope():
            for rgb, infrare in tqdm(zip(rgbs, infrareds)):
                batch = make_batch(rgb, infrare, device=device)
                samples_ddim = inference_model(batch)
                decoded = model.decode_first_stage(samples_ddim)
                output_img = post_process(decoded)
                outpath = os.path.join(opt.outdir, 'sample' + os.path.basename(rgb))
                Image.fromarray(output_img.astype(np.uint8)).save(outpath)

