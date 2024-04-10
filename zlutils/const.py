
#  =============================================== 常量类
class _常量类(type):
    '''常量不能以下划线_开头'''

    def __new__(元类, 类名, 所有基类, 类变量):
        类变量[f'_{类名}__allconsts'] = [变量名 for 变量名 in 类变量.keys() if 变量名[:1]!='_']
        return super().__new__(元类, 类名, 所有基类, 类变量)


    def __setattr__(类, 变量名, 想要设置的值):
        #if 变量名 not in vars(类):
        raise RuntimeError('这是一只固态只因因此不能改变结构')
        #try:
        #    vars(类)[变量名].赋值(想要设置的值)
        #except:
        #    super().__setattr__(变量名, 想要设置的值)

    def __iter__(类):
        return super().__getattribute__(f'_{类.__name__}__allconsts').__iter__()


class 常量类(metaclass=_常量类):

    def __init__(我):
        raise TypeError(f'{我.__class__.__name__}不允许实例化')


# =============================================== 模式类
class _模式类(_常量类):
    '''
应用样例:
from zlutils.const import 模式类
class m(模式类):
    gtt = 0
    add = 0

print(m.gtt)
输出:gtt

'gtt' in m
输出:True

list(m)
输出:['gtt', 'add']

'''

    def __contains__(类, 常量):
        try:
            super().__getattribute__(常量)
            return True
        except:
            return False


    def __getattribute__(类, 常量名):
        取值 = super().__getattribute__(常量名)
        if  '_' == 常量名[:1]:
            return 取值
        return str(常量名)


class 模式类(metaclass=_模式类):
    '''用于区分不同的模式'''
    样例 = 0 # 等于号后的值不生效,可任意设置

    def __init__(我):
        raise TypeError(f'{我.__class__.__name__}不允许实例化')


# =============================================== 字段类
class 字段类:
    pass

