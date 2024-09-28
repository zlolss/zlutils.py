from . import onnx_backend as orb
import numpy as np
import PIL
import PIL.Image


def preprocImage(img: PIL.Image.Image):
    img = img.resize(orb.input_shape[-2:])
    img = np.asarray(img)  # (H,W,3) RGB
    img = img[:, :, ::-1]  # 2 BGR
    img = np.transpose(img, [2, 0, 1])
    # img = img.reshape((1, *img.shape))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    # img = np.ascontiguousarray(img)
    return img


def featurize(img: PIL.Image.Image):
    img = preprocImage(img)
    return orb.run(img)
