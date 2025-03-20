from PIL import Image
import numpy as np
import os
import cv2
import tarfile
import yaml
import pickle
from .path import ensurePath, decimal_to_x64
import time
from io import BytesIO
from .threading import EventThread
from .threading import threading


def numpy_representer(dumper, data):
    if isinstance(data, np.integer):
        return dumper.represent_int(data)
    elif isinstance(data, np.floating):
        return dumper.represent_float(float(data))
    elif isinstance(data, np.bool_):
        return dumper.represent_bool(bool(data))
    else:
        raise TypeError(f"Unrecognized numpy type: {type(data)}")


# 将通用representers添加到yaml的Dumper中
yaml.add_representer(np.integer, numpy_representer)
yaml.add_representer(np.int64, lambda dumper, data: dumper.represent_int(data))
yaml.add_representer(np.floating, numpy_representer)
yaml.add_representer(np.bool_, numpy_representer)
yaml.add_representer(tuple, lambda dumper, data: dumper.represent_list(data))


def imageFormat(image, to_type='image', color='RGB'):
    # 转换图片格式
    # TODO
    pass


def shufflenet205FeatureExtractor(image):
    # input: PIL Image
    # return feature: ndarray, shape(1,n)
    # 初次使用是载入shufflenet
    from .shufflenet import featurize
    return featurize(image).reshape(1, -1)


def imatFuzzyStd(image, kernel_size=3):
    # 利用模糊运算前后差异的标准差衡量原图片清晰程度，数值越大表示图片越清晰
    # 读取图像
    # image = np.array(image.reduce(reduce_times))
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # 使用高斯模糊
    # blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # 高斯滤波
    blurred = cv2.blur(gray, ksize=(kernel_size, kernel_size))  # 均值滤波（最快）
    # 计算原始图像和模糊图像之间的绝对差值
    delta = cv2.absdiff(gray, blurred)
    # 计算标准差作为对比度的估计
    contrast = np.std(delta)
    return float(contrast)


def imatDiffRate(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 1
    diff = cv2.absdiff(a, b)
    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    drop_low_mask = diff > 10  # 应用一个伪激活函数， 也可以在结果中再筛选
    return np.sum(diff[drop_low_mask]) / diff.size / 255


class Frame:
    # 视频中的一个帧除了包含本身画面外，还可以附加时间戳、动态特征、sift特征、关注点、文本说明等其他信息
    # 这些信息用tar和对应类型明文存储有利于移殖
    # 这些信息与帧这一概念是绑定的，且有存取有求，并可能频繁复用，将这些信息封装并统一管理，可以减少下游程序的复杂度
    # 保存的方式为：原始图片+同一路径下的图片名.ext.tar文件
    # usage:
    # setExt(self, name, data, ext_name=None)
    # dump()
    # Frame(src, **kwargs) # src可以是图片路径 或 Frame对象 或 PIL.Image.Image对象
    # TODO 附件中包含另一个frame的处理？避免出现循环
    DEFAULT_DUMP_TYPES = {
        'png': ['motion', 'mask'],
        'jpg': ['rgb'],
        'yaml': ['meta'],
        'npz': ['feature'],
        'pkl': ['callback'],
        }
    DEFAULT_EXT_TYPES = list(DEFAULT_DUMP_TYPES.keys())
    PROPERTIES = ['imat', 'regmat', 'image', 'width', 'height', 'wh', 'hw', 'regsize']

    def __init__(self, src, **kwargs):
        # 通过帧的图片路径或者一个新的画面创建帧，附加信息通过kwargs传递
        # var definition
        self._auto_id = decimal_to_x64(hash(str(time.time())))
        self.auto_name = f'Frame_{self._auto_id}'
        self.__src = src
        self._image_raw = None
        self._imat_raw = None
        self._regmat_raw = None  # 用于图像处理的统一尺度
        # self._thumbnail = None  # TODO 部分处理过程默认缩小图像到指定大小以加速
        self._image_path = None
        self._meta = {}
        self._cache = {}
        self._tmp = {}
        self.__invalid_names = dir(self)
        self._modified = False
        self.regsize = 224
        # create
        if isinstance(src, Image.Image):
            self._image_raw = src
        elif isinstance(src, np.ndarray):  # cv2 rgb mat
            self._imat_raw = src
            '''
            image = Image.fromarray(
                cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
                mode='RGB' if len(src.shape) == 3 else 'L')
            self._image_raw = image
            '''
        elif isinstance(src, str) and os.path.exists(src):
            self._image_path = src.strip()
            meta = self.loadExt('__meta', 'yaml')
            if meta is not None:
                self._meta = meta
        elif isinstance(src, Frame):
            self._image_path = src._image_path
            self._image_raw = src._image_raw
            self._meta = src._meta.copy()
            self._cache = src._cache.copy()
        else:
            raise TypeError(f'[error] Frame src not available {src}')
        for name, data in kwargs.items():
            self.setExt(name, data)
        self.setExt('frame_create_time', time.time())

    @property
    def image(self):
        if self.__src is None:
            return None
        elif self._image_raw is None:
            if self._imat_raw is None:
                self._image_raw = self.loadImage()
            else:
                self._image_raw = Image.fromarray(
                    # cv2.cvtColor(self._imat_raw, cv2.COLOR_BGR2RGB),
                    self._imat_raw,
                    mode='RGB' if len(self._imat_raw.shape) == 3 else 'L')
        return self._image_raw

    @property
    def width(self):
        return self.wh[0]

    @property
    def height(self):
        return self.wh[1]


    @property
    def imat(self):
        if self.__src is None:
            return None
        elif self._imat_raw is None:
            self._imat_raw = np.array(self.image)
        return self._imat_raw

    @property
    def regmat(self):  # 用于图像处理的统一尺度
        if self.__src is None:
            return None
        elif self._regmat_raw is None:
            self._regmat_raw = cv2.resize(self.imat, (self.regsize, self.regsize), interpolation=cv2.INTER_LINEAR)
        return self._regmat_raw



    @property
    def wh(self):
        if self.__src is None:
            return 0, 0
        elif isinstance(self._image_raw, Image.Image):
            return (self._image_raw.width, self._image_raw.height)
        elif isinstance(self._imat_raw, np.ndarray):
            return (self._imat_raw.shape[1], self._imat_raw.shape[0])
        else:
            return self.image.width, self.image.height

    @property
    def hw(self):
        return (self.wh[1], self.wh[0])

    @property
    def path(self):
        return self._image_path

    def _repr_jpeg_(self):
        return self.image._repr_jpeg_()

    def __repr__(self):
        return 'Frame(path:{}, {})'.format(self._image_path, self._meta)

    def __getattribute__(self, key):
        if key.startswith('_') or key not in self._meta:
            return super().__getattribute__(key)
        print(f'[warning] please use getitem method [{key}]')
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self.PROPERTIES:
            return getattr(self, key)
        if key not in self._meta:
            return None
        if key in self._cache:
            return self._cache[key]
        if self._meta[key] in self.DEFAULT_EXT_TYPES:
            cache = self.loadExt(key, self._meta[key])
            if cache is not None:
                self._cache[key] = cache
                return cache
        return self._meta[key]

    def __setitem__(self, key, value):
        self._modified = True
        self.setExt(key, value)

    def __iter__(self):
        return (self.PROPERTIES+list(self._meta.keys())).__iter__()

    @property
    def __ext_path(self):  # 帧的附加信息文件路径
        if self._image_path is not None:
            return self._image_path.strip()+'.ext.tar'

    def get(self, name, default):
        if name in self._meta:
            return self[name]
        return default

    def loadImage(self):
        if os.path.exists(self._image_path):
            return Image.open(self._image_path)
        return None

    def setExt(self, name, data, ext_name=None):
        if name in self.__invalid_names:
            raise KeyError(f'{name} is invalid')
        if ext_name is None:
            if isinstance(data, Image.Image):
                if data.mode == 'L':
                    ext_name = 'png'
                else:
                    ext_name = 'jpg'
            elif isinstance(data, np.ndarray):
                ext_name = 'npz'
            elif type(data).__name__ in ['int', 'float', 'str']:
                pass
            elif type(data).__name__ in ['list', 'dict']:
                if len(data) >0 and type(data[0]) not in ['int', 'float', 'str']:
                    ext_name = 'pkl'
                pass
            elif isinstance(data, Frame):  # Frame 对象只保存路径 TODO 未保存的Frame对象？
                ext_name = data._image_path
            else:
                print(f'{name} is unsupport type {type(data)}: {data}, save as .pkl')
                ext_name = 'pkl'

        if ext_name is not None:
            self._cache[name] = data
            self._meta[name] = ext_name
        else:
            self._meta[name] = data

    def loadExt(self, name, ext_name):
        if not os.path.exists(self.__ext_path):
            return None
        full_name = f'{name}.{ext_name}'
        with tarfile.open(self.__ext_path, 'r') as tar:
            try:
                tarinfo = tar.getmember(full_name)
                # ext_name = name.split('.')[-1]
                with tar.extractfile(tarinfo) as fp:
                    if ext_name == 'yaml':
                        return yaml.load(fp, Loader=yaml.FullLoader)
                    elif ext_name in ['png', 'jpg']:
                        return Image.open(fp)
                    elif ext_name == 'pkl':
                        return pickle.load(fp)
                    elif ext_name == 'npz':
                        return np.load(fp)['data']
                    else:
                        print(f'unsupport type {name} in {self.__ext_path}')
                        return None
            except KeyError:
                print(f'{name} lost in {self.__ext_path}')
                return None

    def dumpExt(self, name, ext_name, data):
        if not os.path.exists(self.__ext_path):
            open_type = 'w:'
        else:
            open_type = 'a'
        if ext_name is None:
            return False
        with tarfile.open(self.__ext_path, open_type) as tar:
            tarinfo = tarfile.TarInfo(name=f'{name}.{ext_name}')
            if ext_name == 'yaml':
                fp = BytesIO(yaml.dump(data).encode('utf-8'))
                tarinfo.size = len(fp.getbuffer())
                tar.addfile(tarinfo, fp)
            elif ext_name == 'jpg':
                image_data = data.tobytes('jpeg', data.mode, 95)
                tarinfo.size = len(image_data)
                tar.addfile(tarinfo, BytesIO(image_data))
            elif ext_name == 'png':
                imat = np.array(data)
                image_data = cv2.imencode('.png', imat)[1].tobytes()
                tarinfo.size = len(image_data)
                tar.addfile(tarinfo, BytesIO(image_data))
            elif ext_name == 'npz':
                fp = BytesIO()
                np.savez(fp, data=data)
                fp.seek(0)
                tarinfo.size = len(fp.getbuffer())
                tar.addfile(tarinfo, fp)
            elif ext_name == 'pkl':
                bytes_data = pickle.dumps(data)
                tarinfo.size = len(bytes_data)
                tar.addfile(tarinfo, BytesIO(bytes_data))
            else:
                print(f'unsupport type {ext_name} of {data}')
        return True

    def loadAllUnset(self):
        # self.image
        for key, ext_name in self._meta.items():
            if ext_name in self.DEFAULT_EXT_TYPES and key not in self._cache:
                cache = self.loadExt(key, ext_name)
                if cache is not None:
                    self._cache[key] = cache

    def dump(self, dir_path=None, file_name=None, free=True):
        # TODO 检查，避免重复写入
        if dir_path is not None:
            dir_path = os.path.abspath(dir_path)
            ensurePath(dir_path)
            if file_name is not None:
                self._image_path = os.path.join(dir_path, f'{file_name}.jpg')
            else:
                self._image_path = os.path.join(dir_path, f'{self.auto_name}.jpg')
        if self._image_path is None:
            temp_path = f'{self.auto_name}.jpg'
            print(f'path error {self._image_path}, switch to {temp_path}')
            self._image_path = temp_path
        if not isinstance(self.image, Image.Image):
            print(f'frame has no image')
            return
        self._image_raw.save(self._image_path)
        self.loadAllUnset()
        # if path is not None:
        #    self.__image_path = path
        # if not os.path.isfile(self.__image_path):

        with tarfile.open(self.__ext_path, 'w:') as tar:
            tarinfo = tarfile.TarInfo(name=f'__Frame__')
            tarinfo.size = 0
            tar.addfile(tarinfo, BytesIO(b'0'))
        for key, ext_name in self._meta.items():
            if ext_name in self.DEFAULT_EXT_TYPES and key in self._cache:
                self.dumpExt(name=key, ext_name=ext_name, data=self._cache[key])
        self.dumpExt(name='__meta', ext_name='yaml', data=self._meta)

        if free:  # 存档后释放内存
            self._cache = {}
            self._image_raw = None

        return self._image_path

    def copy(self):
        return Frame(self)


class FrameShader(EventThread):
    # 生成新的特征并添加到frame中
    # 跳过溢出的帧
    # usage：
    # fs = FrameShader()
    # frame = fs(frame)
    EVENT_FRAME = 1
    _last_frame = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._runing_event = threading.Event()
        self._runing_event.clear()

    @property
    def last_frame(self):
        return self._last_frame

    def forward(self, frame: Frame):
        # 重写
        return Frame(frame)

    def __call__(self, frame):  # TODO 处理过程不应该占用主线程， 用队列
        if self._runing_event.is_set():
            return
        if frame is None:
            return
        self._runing_event.set()
        if not isinstance(frame, Frame):
            frame = Frame(frame)
        frame = self.forward(frame)
        if self._last_frame != frame:
            self._last_frame = frame
            self.send_event(self.EVENT_FRAME, self._last_frame)
        self._runing_event.clear()
        return self._last_frame


class FrameSkipShader(FrameShader):
    # 变化过低的帧
    _reduced_imat = None
    _reduce_times = 2
    _skip_threashold = 0.1
    _last_diff_rate = 1
    _diff_rate_count = 0

    def forward(self, frame: Frame):
        reduced_imat = np.array(frame.image.reduce(self._reduce_times))
        if self._reduced_imat is None:
            self._reduced_imat = reduced_imat
            self._diff_rate_count = 0
            return frame
        elif reduced_imat.shape != self._reduced_imat.shape:
            self._reduced_imat = reduced_imat
            self._diff_rate_count = 0
            return frame
        self._last_diff_rate = imatDiffRate(self._reduced_imat, reduced_imat)
        self._diff_rate_count += self._last_diff_rate
        if self._diff_rate_count < self._skip_threashold:
            return self._last_frame
        self._reduced_imat = reduced_imat
        self._diff_rate_count = 0
        return frame


class MotionShader(FrameShader):
    # 连续输入帧数据，生成一个同帧大小的灰度图添加到帧中，表示当前输入帧相对于一段时间前置帧的运动变化
    # 白色（255）表示刚刚发生的变化，数值越低表示变化发生在越早的时候
    def __init__(
        self,
        decline_per_second=300,  # 每秒衰减的总灰度值
        reduced=2,  # 处理前将图片缩小倍率，精度换速度
        diff_threshold = 50
    ):
        super().__init__()
        self._last_frame = None
        self.decline_per_second = decline_per_second
        self.reduced = int(reduced)
        self.decline_count = 0
        self.diff_threshold = diff_threshold
        self.t0 = time.time()
        self.reset()
        # self.start()

    def reset(self):
        # super().reset()
        self.remain_frame = None
        self.last_player_frame = None

    def resetCount(self, frame_time):
        self.t0 = frame_time
        self.decline_count = 0

    def getCurrentDecline(self, frame_time):
        return int((frame_time - self.t0) * self.decline_per_second - self.decline_count)

    def forward(self, _frame: Frame):
        if self.reduced > 1:
            frame = np.array(_frame.image.reduce(self.reduced))
        else:
            frame = np.array(_frame.image)
        frame_time = _frame.get('timestamp', time.time())
        if self.last_player_frame is None or self.last_player_frame.shape != frame.shape:
            diff_frame = frame
            self.remain_frame = None
        else:
            diff_frame = cv2.absdiff(frame, self.last_player_frame)
        self.last_player_frame = frame
        diff_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        _, bframe = cv2.threshold(diff_frame, self.diff_threshold, 255, cv2.THRESH_BINARY)
        if self.remain_frame is None:
            self.resetCount(frame_time)
            self.remain_frame = bframe
        else:
            decline = self.getCurrentDecline(frame_time)
            self.remain_frame = cv2.max(bframe, cv2.subtract(self.remain_frame, decline, dtype=cv2.CV_8U))
            self.decline_count += decline
        motion_image = Image.fromarray(self.remain_frame)
        if self.reduced > 1:
            motion_image = motion_image.resize(
                (motion_image.width*self.reduced, motion_image.height*self.reduced),
                Image.NEAREST
                )
        _frame['motion_image'] = motion_image  # 修改原frame植入信息
        _frame['motion_imat']  = self.remain_frame
        return _frame
        #return Frame(
        #    _frame,
        #    motion_image=motion_image,
        #    )


def makeFrame(src, **kwargs):
    if src is None:
        return None
    elif isinstance(src, Frame):
        return src
    return Frame(src, **kwargs)
