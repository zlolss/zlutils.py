from .const import _模式类
# todo: 建立独立线程处理触发的事件
import threading, queue


class _事件线程类(threading.Thread):

    def __init__(我, 父类):
        super().__init__(daemon=True)
        我.父类 = 父类
        我.事件队列 = queue.Queue()
        我.睡眠锁 = threading.Condition()
        我.所有监听器 = {}
        # 创建即启动
        我.start()

    def 睡眠(我):
        with 我.睡眠锁:
            我.睡眠锁.wait()

    def 唤醒(我):
        with 我.睡眠锁:
            我.睡眠锁.notify_all()

    def 触发事件(我, 事件, 参数=None):
        我.事件队列.put((事件, 参数))
        #我.唤醒()

    def 处理事件(我, 事件, 参数):
        所有该事件的监听器 = 我.所有监听器.get(事件)
        待移除 = []
        if 所有该事件的监听器 is None:
            return
        for 监听器 in 所有该事件的监听器:
            监听器(参数)
            if 所有该事件的监听器[监听器] == '一次性':
                待移除.append(监听器)
        for 监听器 in 待移除:
            del 所有该事件的监听器[监听器]

    def 添加一次性监听器(我, 事件, 监听器):
        # 触发一次后移除
        if 事件 in 我.所有监听器:
            我.所有监听器[事件][监听器] = '一次性'
        else:
            我.所有监听器[事件] = {监听器:'一次性'}

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

    def run(我):
        while(True):
            #if 我.事件队列.empty():
            #    我.睡眠()
            #    continue
            事件, 参数 = 我.事件队列.get() # 队列的get方法在队列为空时默认阻塞
            我.处理事件(事件, 参数)
            if 我.事件队列.qsize()>5:
                print(f'并发的事件数量{我.事件队列.qsize()}过多, 超过了处理效率.')



class 事件类(metaclass=_模式类):
    创建 = None
    移除 = None

    def __init__(我):
        super().__init__()
        我.__事件线程 = _事件线程类(我)


    def send_event(self, event, params=None):
        return self.__事件线程.触发事件(event, params)

    def 触发事件(我, 事件, 参数=None):
        return 我.__事件线程.触发事件(事件, 参数)

    def add_listener(self, event, callback):
        return self.__事件线程.添加监听器(event, callback)

    def add_listener_once(self, event, callback):
        self.__事件线程.添加一次性监听器(event, callback)

    def 添加监听器(我, 事件, 监听器):
        我.__事件线程.添加监听器(事件, 监听器)

    def 设置唯一监听器(我, 事件, 监听器):
        我.__事件线程.设置唯一监听器(事件, 监听器)

    def remove_listener(self, cls, listener):
        return self.__事件线程.移除监听器(cls, listener)

    def 移除监听器(我, 事件, 监听器):
        return 我.__事件线程.移除监听器(事件, 监听器)

