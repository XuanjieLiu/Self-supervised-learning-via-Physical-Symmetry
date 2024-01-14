from normal_rnn import Conv2dGruConv2d, LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL
from tkinter import *
from PIL import Image, ImageTk
from torchvision.utils import save_image
import os
import sys
from shared import *
from common_utils import load_config


EXP_NAME = sys.argv[1]
MODEL_PATH = sys.argv[2]
CONFIG = load_config(EXP_NAME)
RANGE = 6
TICK_INTERVAL = 0.05
CODE_LEN = CONFIG['latent_code_num']
IMG_PATH = 'vae3DBallEval_ImgBuffer'
IGM_NAME = IMG_PATH + "/test.png"




def init_img_path():
    if not os.path.isdir(IMG_PATH):
        os.mkdir(IMG_PATH)


def init_codes():
    codes = []
    for i in range(0, CODE_LEN):
        codes.append(0.0)
    return codes


def init_vae():
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.eval()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Model is loaded")
    return model


class TestUI:
    def __init__(self):
        init_img_path()
        self.win = Tk()
        self.win.title(os.getcwd().split('\\')[-1])
        self.code = init_codes()
        self.scale_list = self.init_scale_list()
        self.vae = init_vae()
        self.photo = None
        self.label = Label(self.win)
        self.label.pack(side=RIGHT)

    def on_scale_move(self, value, index):
        self.code[index] = float(value)
        self.scale_list[index].set(float(value))
        self.recon_img()
        self.load_img()
        print(self.code)

    def recon_img(self):
        codes = torch.tensor(self.code, dtype=torch.float).to(DEVICE)
        sample = self.vae.decoder(self.vae.fc3(codes).view(1, LAST_CN_NUM, LAST_H, LAST_W))
        save_image(sample.data.view(1, IMG_CHANNEL, sample.size(2), sample.size(3)), IGM_NAME)

    def load_img(self):
        image = Image.open(IGM_NAME)
        img = image.resize((350, 350))
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

    def init_scale_list(self):
        scale_list = []
        for i in range(0, CODE_LEN):
            scale = Scale(
                self.win,
                variable=DoubleVar(value=self.code[i]),
                command=lambda value, index=i: self.on_scale_move(value, index),
                from_=0 - RANGE / 2,
                to=0 + RANGE / 2,
                resolution=0.01,
                length=600,
                tickinterval=TICK_INTERVAL
            )
            scale.pack(side=LEFT)
            scale_list.append(scale)
        return scale_list


if __name__ == "__main__":
    test_ui = TestUI()
    test_ui.win.mainloop()
