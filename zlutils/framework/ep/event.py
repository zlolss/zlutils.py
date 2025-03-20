'''
事件管理系统文档
概述

本事件管理系统用于处理事件的订阅和发布，允许用户为特定事件类型注册回调函数，并在事件触发时按优先级执行这些函数。系统主要由Event类和EventManager类组成。
类定义
Event类

Event类用于表示一个事件，包含事件名称和相关数据。
属性

    name: 事件的名称。
    data: 与事件相关的数据。

方法

    __init__(self, name, data): 构造函数，初始化事件名称和数据。

EventManager类

EventManager类负责管理事件的订阅和发布。
属性

    listeners: 一个字典，键为事件类型，值为一个列表，列表中包含元组（回调函数，优先级）。

方法

    __init__(self): 构造函数，初始化listeners字典。
    subscribe(self, eventtype, callback, priority): 订阅事件，将回调函数添加到指定事件类型的列表中，根据优先级排序。
        eventtype: 事件类型。
        callback: 回调函数。注: callback应该有阻塞和非阻塞的执行方式
        priority: 回调函数的优先级，数值越小优先级越高。
    publish(self, eventtype, data): 发布事件，触发指定事件类型的所有回调函数，按照优先级顺序执行。
        eventtype: 事件类型。
        data: 传递给回调函数的数据。
'''

import bisect


class Event:
    def __init__(self, event_type, data=None):
        self.type = event_type
        self.data = data

    def __repr__(self):
        return f'<Event "{self.type}", data={self.data}>'


class EventManager:
    def __init__(self):
        # TODO 向上和向下跨阶级传递
        self.listeners = {}

    def subscribe(self, event_type, callback, priority=-1):
        if priority is None:
            priority = -1

        if event_type not in self.listeners:
            self.listeners[event_type] = []

        if len(self.listeners[event_type]) == 0:
            priority = max(priority, 0)
            self.listeners[event_type].append((callback, priority))
        elif priority == 0:
            self.listeners[event_type].insert(0, (callback, priority))
        elif priority == -1:
            priority = max(self.listeners[event_type], key=lambda x:x[1])
            self.listeners[event_type].append((callback, priority))
        else:
            i = bisect.bisect_right(self.listeners[event_type], priority, key=lambda x: x[1])
            self.listeners[event_type].insert(i, (callback, priority))

    def publish(self, event_type, event_data=None):
        e = Event(event_type, event_data)
        if event_type in self.listeners:
            for callback, _ in self.listeners[event_type]:
                callback(e)


if __name__ == "__main__":

    def high_priority_callback(e):
        print("High priority callback:", e.data)

    def medium_priority_callback(e):
        print("Medium priority callback:", e.data)

    def low_priority_callback(e):
        print("Low priority callback:", e.data)

    def lowest_priority_callback(e):
        print("Lowest priority callback:", e.data)

    event_manager = EventManager()
    event_manager.subscribe('myevent', high_priority_callback, priority=1)
    event_manager.subscribe('myevent', medium_priority_callback, priority=10)
    event_manager.subscribe('myevent', low_priority_callback, priority=20)
    event_manager.subscribe('myevent', lowest_priority_callback, priority=-1)

    # print(event_manager.listeners)
    # 触发事件
    event_manager.publish('myevent', "Hello, World!")
