
'''
event_type中的路径用`.`隔开
故event_type的实际名称中不能包含`.`

PluginManager实例化时自动载入Plugins目录下的插件

插件中必须包含CREATORS列表列出需要创建的插件类
例如:
CREATORS = [Plugin]

'''

# if __name__ == "__main__":
#    from event_manager import EventManager
# else:
#    from .event_manager import EventManager


import importlib
import pkgutil
if __name__ == "__main__":
    from config import ConfigManager
else:
    from .config import ConfigManager


class Plugin:
    # 插件模板
    # --- 插件重载方法 ---
    def initVar(self):
        pass

    def initEvent(self):
        # 初始化过程中用self.subscribeEvent(event_type, callback, priority=-1)注册事件
        pass

    # --- 初始化 ---
    def __init__(self, plugin_manager=None):
        self.__pm = plugin_manager  # 插件管理器, 不可更改
        self.conf = ConfigManager(section=self._plugin_tag)
        if self.__pm is None:
            if __name__ == "__main__":  # debug
                from event import EventManager
            else:
                from .event import EventManager
            self._event_manager = EventManager()
        else:
            self._event_manager = None
            self.conf._init_config_parser = self.__pm.conf._init_config_parser
        self._plugins = {}
        # 预处理
        self.initVar()
        self.initEvent()
        # self.publishEvent('initialized')

    # TODO --- 插件管理 ---

    def addPlugin(self, plugin_creator):
        new_plugin = plugin_creator(plugin_manager=self)
        # print(plugin_creator.__bases__)
        if __name__ != "__main__":  # debug
            assert isinstance(new_plugin, Plugin)
        assert new_plugin._plugin_tag not in self._plugins
        self._plugins[new_plugin._plugin_tag] = new_plugin
        return new_plugin._plugin_tag

    def _loadPlugins(self, plugin_dir):
        package = importlib.import_module(plugin_dir)
        for _, name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{plugin_dir}.{name}")
            # print(module, getattr(module, 'CREATORS'))
            creators = getattr(module, 'CREATORS')
            for creator in creators:
                self.addPlugin(creator)
        # plugin_class = getattr(plugin_module, 'Plugin')
        #plugin_instance = plugin_class(self.event_manager)
        #self.lifecycle_manager.register_plugin(plugin_path, plugin_instance)
        #self.lifecycle_manager.init_plugins()

    '''def unloadPlugin(self, plugin_name):
        self.lifecycle_manager.stop_plugin(plugin_name)
        self.lifecycle_manager.destroy_plugins()
        del self.lifecycle_manager.plugins[plugin_name]'''

    # --- 事件管理 ---

    @property
    def _plugin_tag(self):
        return str(self.__class__.__name__).lower()

    def wrapEventType(self, event_type):
        # 避免重复封装
        if event_type.startswith('.'):
            pass
        elif self.__pm is not None:
            event_paths = event_type.lower().split('.')
            event_paths = list(filter(lambda x: len(x) > 0, event_paths))
            event_type = '.'.join(event_paths)
            if self._plugin_tag in event_paths:
                event_type = '.' + event_type
            else:
                event_type = f'.{self._plugin_tag}.' + event_type
        else:
            event_type = '.' + event_type
        return event_type.lower()

    def publishEvent(self, event_type, event_data=None, source=None):
        event_type = self.wrapEventType(event_type)
        source = self if source is None else source
        # print(event_type)
        if self.__pm is not None:
            self.__pm.publishEvent(event_type, event_data, source=source)
        else:
            self._event_manager.publish(event_type, event_data, source=source)

    def subscribeEvent(self, event_type, callback, priority=None):
        if priority is None and '.' not in event_type:
            priority = 0
        event_type = self.wrapEventType(event_type)
        if self.__pm is not None:
            self.__pm.subscribeEvent(event_type, callback, priority)
        else:
            self._event_manager.subscribe(event_type, callback, priority)


class PluginManager(Plugin):

    def __init__(self, plugin_dir='Plugins', config_path='config.ini'):
        super().__init__(plugin_manager=None)
        self.config_path = config_path
        self.conf._section = 'main'
        self.conf.loadINI(config_path)
        self.plugin_dir = plugin_dir
        self._loadPlugins(plugin_dir=plugin_dir)


# 通过CREATORS创建插件
CREATORS = [Plugin]

if __name__ == "__main__":
    class Plugin_a(Plugin):
        def initEvent(self):
            self.subscribeEvent('.pmprint', lambda e: print(f'._a, {e}') or 0)
            self.subscribeEvent('.pmprint2', lambda e: print(f'._a, {e}') or 0)
            self.subscribeEvent('.Plugin_b.pmprint', lambda e: print(f'._a, {e}') or 0)
        pass

    class Plugin_b(Plugin):
        def initEvent(self):
            self.subscribeEvent('.pmprint', lambda e: print(f'._b, {e}') or 0)
            self.subscribeEvent('pmprint', lambda e: print(f'._bb, {e}') or 0)
        pass

    pm = PluginManager()
    pa = Plugin_a(plugin_manager=pm)
    pb = Plugin_b(plugin_manager=pm)
    pm.publishEvent('pmprint')
    pm.publishEvent('.pmprint')
    pm.publishEvent('pmprint2')
    pb.publishEvent('pmprint')
    print(pm._event_manager.listeners)
    print(pm.conf._init_config_parser)
    print(pm.conf._init_config_parser.get('main', 'test'))
    print(type(pm.conf._init_config_parser.get('main', 'test')))
    # print(pm.conf._init_config_parser.get('main1', 'test'))
    print(pm.conf._init_config_parser.get('main', 'test1', fallback='mone'))
    print(pm.conf.test)
    pm.conf._default_config['test'] = 2
    print(pm.conf.test)
    pm.conf._manual_config['test1'] = 3
    print(pm.conf.test1)
    print(pm.conf.test3)
