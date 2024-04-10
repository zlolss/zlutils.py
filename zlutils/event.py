from .const import _模式类
# todo: 建立独立线程处理触发的事件



class 事件类(metaclass=_模式类):
    创建 = None
    移除 = None

    def __init__(我):
        super().__init__()
        我.所有监听器 = {}

    def sendevent(我, event, **params):
        return 我.触发事件(event, **params)

    def 触发事件(我, 事件, **参数):
        所有该事件的监听器 = 我.所有监听器.get(事件)
        if 所有该事件的监听器 is None:
            return
        for 监听器 in 所有该事件的监听器:
            监听器(参数)

    def 添加监听器(我, 事件, 监听器):
        if 事件 in 我.所有监听器:
            我.所有监听器[事件][监听器] = None
        else:
            我.所有监听器[事件] = {监听器:None}

    def 设置唯一监听器(我, 事件, 监听器):
        我.所有监听器[事件] = [监听器]

    def 移除监听器(我, 事件, 监听器):
        if 事件 in 我.所有监听器 and 监听器 in 我.所有监听器[事件]:
            del 我.所有监听器[事件][监听器]
