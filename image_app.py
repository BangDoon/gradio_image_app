import gradio as gr
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import cv2
from torchvision import transforms
import os

model = PRN_r(6, 1)

model = model.cuda()
model.load_state_dict(torch.load('./net_latest.pth'))
iscuda = torch.cuda.is_available()

data_path = './datasets/Rain100H/'
file_list = [data_path+file_name for file_name in os.listdir(data_path)]


def deNosing(noise_img):
    
    y = np.array(noise_img)
    y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = Variable(torch.Tensor(y))

    if iscuda:
        y = y.cuda()

    with torch.no_grad():
        if iscuda:
            torch.cuda.synchronize()

        out, _ = model(y)
        out = torch.clamp(out, 0., 1.)

        if iscuda:
            torch.cuda.synchronize()


    if iscuda:
        save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
    else:
        save_out = np.uint8(255 * out.data.numpy().squeeze())

    save_out = save_out.transpose(1, 2, 0)
    save_out = cv2.resize(save_out,(noise_img.width,noise_img.height),interpolation=cv2.INTER_CUBIC)
    return save_out

demo = gr.Interface(deNosing,
                    gr.inputs.Image(type='pil'),
                    gr.outputs.Image(type='pil'),
                    examples=file_list
                    )

demo.launch(debug=True, share=True)
