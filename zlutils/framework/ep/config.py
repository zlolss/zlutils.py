# config
# - 优先使用定义值, 其次使用默认值
# - 用便捷使用.方法获取数值

import configparser


class ConfigManager:
    def __init__(self, section='main', init_config_parser=configparser.ConfigParser()):
        # 生效优先级从下到上
        self._section = section
        self._default_config = {}  # 默认值定义
        self._rem = {}  # 注释
        self._init_config_parser = init_config_parser  # 生成时赋予
        self._manual_config = {}  # 后续更改
        pass

    def loadINI(self, cpath):
        try:
            conf = configparser.ConfigParser()
            conf.read(cpath)
            self._init_config_parser = conf
        except Exception as e:
            pass

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except Exception as e:
            name = name.lower()  # 全部用小写
            print(self._section, name)
            if name in self._manual_config:
                return self._manual_config[name]
            elif self._init_config_parser.has_option(self._section, name):
                raw_str = self._init_config_parser.get(self._section, name, raw=False)
                if name in self._default_config:
                    return type(self._default_config[name])(raw_str)
                return raw_str
            elif name in self._default_config:
                return self._default_config[name]
            raise AttributeError(name)

    def setDefault(self, name, value, rem=None):
        self._default_config[name] = value
        if rem is not None:
            self._rem[name] = rem

    def set(self, name, value):
        self._manual_config[name] = value
