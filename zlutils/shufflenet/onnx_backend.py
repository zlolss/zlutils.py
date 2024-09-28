import onnxruntime as ort
import numpy as np
import os
import time
# import cv2
_curdir = os.path.dirname(os.path.abspath(__file__))

print('shufflenetv2 session loading...', end='')
t0 = time.time()
session = ort.InferenceSession(os.path.join(_curdir, "shufflenetv2_0.5x_nofc_qint8.onnx"))
print(f'ok.(cost {((time.time()-t0)*1000):.3f}ms)')
input_name = session.get_inputs()[0].name  # 获取输入层的名称
input_shape = session.get_inputs()[0].shape[1:]  # 获取输入层的形状


def run(input_data: np.ndarray):
    return session.run(None, {input_name: input_data})[0]
