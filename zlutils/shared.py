'''
v0.1 2024-03-22
'''

import logging, time
日志 = logging.getLogger()

_默认字符串编码 = 'utf-8'
_允许的数据类型 = [int, float, str]


class _共享列表映射:
        def __init__(我, 共享列表, 列表索引):
            我.列表索引 = 列表索引
            我.共享列表 = 共享列表
        def 取值(我):
            return 我.共享列表[我.列表索引]
        def __get__(我, 实例, 甲方):
            return 我.取值()
        def 赋值(我, 值):
            if type(值) not in _允许的数据类型:
                raise RuntimeError(f"赋值类型{type(值)}不在许可范围({_允许的数据类型})中")
            我.共享列表[我.列表索引] = 值
        def __repr__(我):
            return f'{我.__class__.__name__}({我.取值()})'


class _只因(type):

    心 = 0.0

    def __new__(元类, 类名, 所有基类, 类变量):
        if 类名=='公鸡':
            return super().__new__(元类, 类名, 所有基类, 类变量)
        import base64

        共享列表名 = 'zls_'+ base64.b64encode(类名.encode(_默认字符串编码)).decode(_默认字符串编码)
        变量名列表 = [变量名 for 变量名 in 类变量 if (type(类变量[变量名]) in _允许的数据类型)and(变量名[:1]!='_')]
        类变量_元类 = vars(元类)
        变量名列表_元类 = [变量名 for 变量名 in 类变量_元类 if (type(类变量_元类[变量名]) in _允许的数据类型)and(变量名[:1]!='_')]
        变量名列表_合成 = 变量名列表 + 变量名列表_元类
        from multiprocessing import shared_memory
        try:
            共享列表 = shared_memory.ShareableList( name=共享列表名 )
            日志.info("载入共享列表, 覆盖默认值")
        except:
            值列表 = [ 类变量[变量名] for 变量名 in 变量名列表 ]
            值列表_元类 = [类变量_元类[变量名] for 变量名 in 变量名列表_元类]
            值列表_合成 = 值列表+值列表_元类
            共享列表 = shared_memory.ShareableList( sequence = 值列表_合成, name=共享列表名 )
            日志.info("以默认值创建共享列表")
        类变量[f'_{类名}___共享列表'] = 共享列表
        for 索引值, 变量名 in enumerate(变量名列表_合成):
            类变量[变量名] = _共享列表映射(共享列表, 索引值)
        return super().__new__(元类, 类名, 所有基类, 类变量)

    def __setattr__(类, 变量名, 想要设置的值):
        if 变量名 not in vars(类):
            raise RuntimeError('这是一只固态的只因不能改变结构')
        try:
            vars(类)[变量名].赋值(想要设置的值)
        except:
            super().__setattr__(变量名, 想要设置的值)


import threading
class 公鸡(metaclass=_只因):
    '''使用说明:
派生类类中的类变量将通过内存共享给本环境中的所有的python程序.
以派生类的名称为唯一标识.
类变量值仅支持基本数据类型:int,float,str.
_开头的类变量会被忽略
已存在的赋值会覆盖默认值
不支持自定义方法
当所有类被释放后共享内存将被释放

派生类定义模板如下:

class 不会下蛋的( 公鸡 ):
    描述 = "默认值"
    食量 = 0

# 赋值
不会下蛋的.食量 = 1
# 取值
print(不会下蛋的.食量)
'''

    心 = 0.0

    def __new__(cls, *_, **__):
        raise RuntimeError("冷知识: 公鸡不能自我繁殖")


class 鸡心管理器:

    def __init__(我, 一只只因, 标准心律=120):
        我.给自己绑定一只只因(一只只因)
        我.标准心律 = 标准心律

    def 给自己绑定一只只因(我, 一只只因):
        if isinstance(一只只因, _只因):
            我.只因 = 一只只因
        else:
            try:
                _ = 一只只因.心
                一只只因.心 = time.time()
                我.只因 = 一只只因
                日志.warning('虽然不是只因, 但勉强能用吧?...大概')
            except:
                日志.error('没有心吗?(꒪⌓꒪)')
                raise RuntimeError(f"你确定这玩意儿({一只只因})是只因?")

    @property
    def 心跳间隔(我):
        return 1 / 我.标准心律

    @property
    def 心跳检测间隔(我):
        return 1.5 / 我.标准心律

    def 跳一下(我):
        我.只因.心 = time.time()

    @property
    def 有心跳(我):
        return (time.time()-我.只因.心) <= 我.心跳检测间隔

    def 一直跳(我):
        我.跳一下()
        我.准备跳 = threading.Timer(我.心跳间隔 ,我.一直跳)
        我.准备跳.start()

    def 一颗成熟的心要学会自己跳(我):
        我.准备跳 = threading.Timer(我.心跳间隔 ,我.一直跳)
        我.准备跳.start()

    def 似了(我):
        我.准备跳.cancel()

    def 复活(我):
        我.准备跳 = threading.Timer(我.心跳间隔 ,我.一直跳)
        我.准备跳.start()

    def 这次真的似了(我):
        我.准备跳.cancel()









