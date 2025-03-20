
import threading
import logging
import traceback
import time
from collections import OrderedDict
from queue import Queue
import typing


class Player:
    '''按fps执行_Player_Step
    包含play, pause, stop功能'''

    def __init__(self, fps=30):
        self._player_play = False
        self._player_fps = fps
        self._player_thread = None
        pass

    def stop(self):
        self._Player_Pause()

    def _Player_Play(self):
        if self._player_is_playing:
            print('[warn] old player_thread.is_alive() ')
            self._player_play = True
            return
        self._player_play = True
        self._player_thread = threading.Thread(target=self._Player_Run)
        self._player_thread.start()
        pass

    def _Player_Pause(self):
        self._player_play = False
        pass

    def _Player_Reset(self):
        # TODO
        pass

    @property
    def _player_is_playing(self):
        return self._player_thread is not None and self._player_thread.is_alive()

    def _Player_Run(self):
        base_time = None
        step_count = 0
        while self._player_play:
            if base_time is None or step_count > 0xffff:
                base_time = time.time()
                step_count = 0

            self._Player_Step()
            step_count += 1
            wait_time = (step_count+1) / self._player_fps - (time.time() - base_time)
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                base_time = time.time()
                step_count = 0

    def _Player_Step(self):
        # 重载
        pass

    class _Source_Event:
        # 用于管理帧更新
        def __init__(self):
            self._last = None
            self._flag = False

        def setdata(self, data):
            if id(data) != id(self._last):
                self._last = data
                self._flag = True

        def done(self):
            self._flag = False

        @property
        def data(self):
            return self._last

        @property
        def undo(self):
            return self._flag


class Restartable:
    # 可重启的线程类

    def __init__(
        self,
        target=None,  # target, target_step 二选一
        target_step=None,  # target, target_step 二选一
        kwargs={},
        error_event=threading.Event(),  # 可以从外部绑定event用于监听
        stop_event=threading.Event(),  # 可以从外部绑定event用于监听
    ):
        self.__thread: threading.Thread = None
        self.__target = target
        self.__target_step = target_step
        self.__kwargs = kwargs
        self.__error_event = error_event
        self.__stop_event = stop_event
        self.__running = False
        self.__run_count = 0

    def is_alive(self):
        return self.__running

    def runstep(self):
        # 按步骤 循环执行 重写这部分
        if self.__target_step is not None:
            self.__target_step(**self.__kwargs)
            return
        self.__running = False
        pass

    def run(self):
        # 一次性执行 重写这部分
        try:
            self.__error_event.clear()
            self.__stop_event.clear()
            self.__running = True
            self.__run_count += 1
            print(f'{self.__running}运行{self.__run_count}次')
            if self.__target is not None:
                self.__target(**self.__kwargs)
            else:
                while self.__running:
                    self.runstep()
        except Exception as e:
            logging.error(f'{e}\n{traceback.format_exc()}')  # 输出完整错误日志
            self.__error_event.set()
        finally:
            self.__running = False
            self.__stop_event.set()

    def stop(self):  # 此操作会产生阻塞
        self.__running = False
        if self.__thread.is_alive():
            self.__thread.join()
        self.__stop_event.set()

    def start(self):
        if self.__thread and self.__thread.is_alive():
            # already run
            return
        self.__thread = threading.Thread(target=self.run)
        self.__running = True
        self.__thread.start()

    def restart(self):  # 此操作会产生阻塞
        self.stop()
        self.start()

    def kill(self):
        return self.stop()


class OverloadSkip:
    # 用于封装方法，调用时如果被占用则跳过，用于实时流处理
    def __init__(self, target=lambda: print('blank'), max_fps=30):
        self._target = target
        self._thread = None
        self._last_call_time = 0
        self._max_fps = max_fps

    @property
    def _call_interval(self):
        if self._max_fps <= 0:
            return 0
        return 1/self._max_fps

    def __call__(self, *args, **kwargs):
        if self._thread is not None and self._thread.is_alive():
            return
        if time.time() - self._last_call_time < self._call_interval:
            return
        self._thread = threading.Thread(target=self._target, args=args, kwargs=kwargs)
        self._last_call_time = time.time()
        self._thread.start()


class Supervised:  # 自重启服务
    def __init__(
        self,
        target=None,  # target, target_step 二选一
        target_step=None,  # target, target_step 二选一
        restart_interval=1,
        error_event=threading.Event(),  # 可以从外部绑定event用于监听
        stop_event=threading.Event(),
        switch_event=threading.Event(),
        **kwargs
    ):
        self.__target = target
        self.__target_step = target_step
        self._kwargs = kwargs
        self.__thread = None
        self.__enabled = True  # 自动restart
        self.__switch_event = switch_event
        self.__killed = False
        self.__restart_interval = restart_interval
        self.__stop_event = stop_event
        self.__restart_count = 0

    def is_enabled(self):
        return self.__enabled

    def is_alive(self):
        return self.__thread.is_alive()

    def check_killed(self):
        if self.__killed:
            logging.error(f'supervisor killed:\n{traceback.format_exc()}')
        return self.__killed

    def setEnable(self):  # 取代 start 用以启动内部线程
        self.check_killed()
        self.__enabled = True
        self.__switch_event.set()

    def setDisable(self):  # 取代 stop 用以停止内部线程
        self.check_killed()
        self.__enabled = False
        self.stopThread()
        self.__switch_event.set()

    def kill(self):  # 完全关闭 不可重启
        self.setDisable()
        self.__killed = True
        self.stopThread()
        self.setEnable()

    def stopThread(self):
        self.__running = False
        if self.__thread.is_alive():
            self.__stop_event.wait()

    def runstep(self):
        # 循环执行重写
        if self.__target_step is not None:
            self.__target_step(**self.__kwargs)
            return
        print('heart')
        time.sleep(1)
        pass

    def run(self):
        # 单次任务重写
        if self.__target is not None:
            self.__target(**self.__kwargs)
            return

        while self.__running:
            self.runstep()
        # self.setDisable()
        pass

    def supervisedRun(self):
        while not self.__killed:
            if not self.__enabled:
                self.__switch_event.wait()
                if self.__killed:
                    break
                elif not self.__enabled:
                    continue
            try:
                self.__restart_count += 1
                self.__running = True
                self.run()
                self.setDisable()  # 正常完成任务
            except Exception as e:
                logging.error(f'{e}\n{traceback.format_exc()}')
            finally:
                self.__running = False
                self.__stop_event.set()
            time.sleep(self.__restart_interval)

    def start(self):
        if self.__thread and self.__thread.is_alive():
            # already run
            return
        self.__thread = threading.Thread(target=self.supervisedRun)
        self.__running = True
        self.__thread.start()

    def stop(self):
        self.stopThread()

    def join(self):
        self.__thread.join()


class _EventThread(threading.Thread):
    def __init__(self, parent=None, daemon=True, **kwargs):
        super().__init__(daemon=daemon, **kwargs)
        self._parent = parent
        self._event_queue = Queue()
        self._event_lock = threading.Condition()
        self._event_listeners = {}

    def add_listener(self, event, listener):
        if event in self._event_listeners:
            self._event_listeners[event][listener] = None
        else:
            self._event_listeners[event] = {listener: None}

    def send_event(self, event, *args, **kwargs):
        self._event_queue.put((event, args, kwargs))

    def remove_listener(self, event, listener):
        listeners = self._event_listeners.get(event)
        if listeners is None:
            return
        if listener in listeners:
            del listeners[listener]

    def stop(self):
        self._event_queue.put(None)

    def run(self):
        while True:
            # print('_EventThread start')
            pac = self._event_queue.get()
            if pac is None:
                break
            event, args, kwargs = pac
            listeners = self._event_listeners.get(event)
            # print(listeners)
            if listeners is None:
                # print(f'Event {event} has no listener')
                continue
            for listener in listeners:
                listener(*args, **kwargs)


class 异步处理单元(Supervised):
    '''用于在帧特征处理过程中由于处理步骤的复杂度不同应该允许跳帧以避免卡顿
当存在多个供应商时,其中一个上新,就需要处理单元重新生产
'''

    def __init__(self, 供应商列表={}, 最大帧率=-1, **附加参数):
        super().__init__(**附加参数)
        self._产出通知列表 = set()
        self.时间戳 = 0
        self.产品 = None

        self.最大帧率 = 最大帧率
        self.附加参数 = 附加参数

        self._原料通知 = threading.Condition()
        self._供应商列表 = {}  # {产品名:供应商}
        self.添加供应商(**供应商列表)

        # self.__上一次生产完成时间 = 0
        self.后初始化()

    @property
    def 生产间隔(self):
        return 1/self.最大帧率 if self.最大帧率 > 0 else 0

    def 后初始化(self):
        pass

    def 添加供应商(self, **供应商列表):
        合法供应商 = {名称: 供应商 for 名称, 供应商 in 供应商列表.items() if isinstance(供应商, 异步处理单元)}
        附加参数追加 = {名称: 供应商 for 名称, 供应商 in 供应商列表.items() if not isinstance(供应商, 异步处理单元)}
        for 名称 in 附加参数追加:
            if 名称 not in self.附加参数:
                self.附加参数[名称] = 附加参数追加[名称]
            else:
                print(f'重复定义{名称}')

        self._供应商列表.update(合法供应商)
        for 供应商 in self._供应商列表.values():
            供应商._产出通知列表.add(self._原料通知)

    def _生产完成(self):
        for 通知 in self._产出通知列表:
            with 通知:
                通知.notify_all()

    def 回调_原料通知(self, *args, **kwargs):
        with self._原料通知:
            self._原料通知.notify_all()

    def _等待原料(self):
        with self._原料通知:
            self._原料通知.wait()
        剩余时间 = self.生产间隔 - (time.time() - self.时间戳)
        if 剩余时间 > 0:
            time.sleep(剩余时间)
        # self.时间戳 = time.time()

    def 生产(self, **原料):
        # 重写这部分, 返回产品
        pass

    def 启动(self):
        return self.start()

    def 启用(self):
        return self.setEnable()

    def 禁用(self):
        return self.setDisable()

    def 终止(self):
        return self.kill()

    def runstep(self):
        # 循环执行重写
        self._等待原料()
        产品 = self.生产(**{
            产品名: 供应商.产品 for 产品名, 供应商 in self._供应商列表.items()
                })
        if 产品 is None:
            return
        self.时间戳 = time.time()
        self.产品 = 产品
        self._生产完成()


class 异步管理器:

    @property
    def 所有异步单元(self):
        return {
            单元名称: 单元
            for 单元名称, 单元 in self.__dict__.items()
            if not 单元名称.startswith('_') and isinstance(单元, 异步处理单元)
            }

    def 启动(self):
        for 单元名称, 单元 in self.所有异步单元.items():
            单元.启动()

    def 暂停(self):
        for 单元名称, 单元 in self.所有异步单元.items():
            单元.禁用()

    def 继续(self):
        for 单元名称, 单元 in self.所有异步单元.items():
            单元.启用()

    def 终止(self):
        for 单元名称, 单元 in self.所有异步单元.items():
            单元.终止()


class ThreadManager(threading.Thread):
    # 统一管理线程启停
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t = OrderedDict()
        # self.__internal_names = dir(self)

    def _checkThread(self, t):
        attrs = ['start', 'stop', 'is_alive', 'join']
        for attr in attrs:
            if not hasattr(t, attr):
                return False
        return True

    def __setattr__(self, name, value):
        if name.startswith('_') or name in dir(self):
            return super().__setattr__(name, value)
        if self._checkThread(value) and name not in self._t:
            self._t[name] = value
        else:
            return super().__setattr__(name, value)
            # raise RuntimeError(f'{name}:{value} not a thread')

    def __getattribute__(self, name):
        if name.startswith('_') or name in dir(self):
            return super().__getattribute__(name)
        return self._t[name]

    def run(self):
        # 主线程重写
        pass

    def is_alive(self):
        result = False
        for name, t in self._t.items():
            result = result or t.is_alive()
        result = result or super().is_alive()
        return result

    def start(self):
        for name, t in self._t.items():
            if not t.is_alive():
                print(f'start: {name}')
                t.start()
        super().start()

    def stop(self):
        for name, t in self._t.items():
            if t.is_alive():
                print(f'stop: {name}')
                t.stop()
        # super().stop()

    def batchRun(self, func_name, *args, **kwargs):
        # 批量执行
        for name, t in self._t.items():
            if hasattr(t, func_name) and isinstance(getattr(t, func_name), typing.Callable):
                getattr(t, func_name)(*args, **kwargs)

    def join(self):
        for name, t in self._t.items():
            t.join()
        super().join()


class EventThread(ThreadManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t['event_thread__'] = _EventThread()

    def add_listener(self, event, listener):
        return self.event_thread__.add_listener(event, listener)

    def send_event(self, event, *args, **kwargs):
        # print(f'send {event} {args}')
        self.event_thread__.send_event(event, *args, **kwargs)

    def remove_listener(self, event, listener):
        self.event_thread__.remove_listener(event, listener)

    def run(self):
        self.event_thread__.join()


if __name__ == "__main__":
    rt = Restartable()
    for i in range(3):
        rt.start()
