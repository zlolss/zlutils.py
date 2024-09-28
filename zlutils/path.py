import os
import re
import inspect


def decimal_to_x64(num):
    if num == 0:
        return '0'

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()"
    flag = "0" if num > 0 else '1'
    x64 = ''
    num = abs(num)

    while num > 0:
        remainder = num % 64
        x64 = digits[remainder] + x64
        num //= 64

    return flag + x64


def ensurePath(path):
    if not os.path.exists(path):
        # 如果路径不存在，则创建它
        os.makedirs(path, exist_ok=True)
    return path


def sanitizeFilename(filename):
    # 使用正则表达式替换掉非法字符
    illegal_chars = r'[<>:"\|\?\*]'
    sanitized_filename = re.sub(illegal_chars, '_', filename)
    return sanitized_filename


def getDefaultDataDir():
    home_dir = os.path.expanduser('~')
    return os.path.join(home_dir, 'zldata')


def getDefaultDir(dir_name=None):
    # dir_name 为None时对不同脚本自动生成不同的文件夹路径
    if dir_name is None:
        caller_path = inspect.stack()[1].filename
        script_name = os.path.basename(caller_path).split('.')[0]
        dir_name = f'{script_name}'
    dir_name = sanitizeFilename(dir_name)
    dir_path = os.path.join(getDefaultDataDir(), dir_name)
    # dir_path = ensurePath(dir_path)
    return dir_path


def getPrivateDir(dir_name=None):
    # dir_name 为None时对不同脚本自动生成不同的文件夹路径
    if dir_name is None:
        caller_path = inspect.stack()[1].filename
        script_name = os.path.basename(caller_path).split('.')[0]
        dir_name = f'{script_name}_{decimal_to_x64(hash(caller_path))}'
    dir_name = sanitizeFilename(dir_name)
    dir_path = os.path.join(getDefaultDataDir(), dir_name)
    # dir_path = ensurePath(dir_path)
    return dir_path


class AutoPath:
    # 自动生成存储路径 dump_dir以及dump_name,提供方法getPath(name, ext)
    def __init__(self, name=None, dump_dir=None):
        self._dump_dir = dump_dir
        self._dump_name = name
        self._auto_id = decimal_to_x64(hash(str(inspect.stack()[1][4])))
        self._auto_name = f'{self.__class__.__name__}_{self._auto_id}'
        self._auto_dump_dir = getDefaultDir(self.dump_name)

    @property
    def dump_dir(self):
        tmp = self._dump_dir
        if tmp is None:
            tmp = self._auto_dump_dir
        tmp = sanitizeFilename(tmp)
        # tmp = os.path.join(tmp, self.dump_name)
        return os.path.abspath(tmp)

    @property
    def dump_name(self):
        tmp = self._dump_name
        if tmp is None:
            tmp = self._auto_name
        tmp = sanitizeFilename(tmp)
        return tmp

    def getPath(self, name=None, ext=None):
        if name is None:
            name = self.dump_name
        if ext is None:
            ext = 'default'
        ext = ext.replace('.', '')
        filename = f'{name}.{ext}'
        return os.path.join(self.dump_dir, filename)

