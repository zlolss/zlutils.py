from zlutils.framework.ep.plugin import Plugin


class Plugin_Example(Plugin):

    def initEvent(self):
        # 初始化过程中用self.subscribeEvent(event_type, callback, priority=-1)注册事件
        # 自身的事件类型不加前缀
        # 外部事件需要加上前缀'.'
        self.subscribeEvent('.pmprint', self.cb_pmprint)
        self.subscribeEvent('.pmprint_b', self.cb_pmprint)
        pass

    def cb_pmprint(self, e):
        print(self._plugin_tag, e)


# 通过CREATORS创建插件
CREATORS = [Plugin_Example]
