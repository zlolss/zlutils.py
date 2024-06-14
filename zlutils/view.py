
单元显示长度限制 = 128
深度限制 = 10

def 以文本显示变量结构_深度缩进(变量, 深度=0): # 递归调用
    缩进 = ' '*深度
    if 深度>深度限制:
        return ' '*深度限制 +f"...达到深度限制{深度限制}，下面的内容无法显示..."
    try:
        变量类型 = getattr(type(变量),'__name__')
        if 变量类型 in ['int', 'float', 'bool', 'str']:
            显示内容 = f'{str(变量)}:::{变量类型}'
            if len(显示内容)>单元显示长度限制:
                显示内容 =  f'{显示内容[:单元显示长度限制]}...:::{变量类型}'
            else:
                显示内容 = f'{显示内容}:::{变量类型}'
        elif 变量类型 == 'ndarray':
            显示内容 = f'{变量类型},矩阵, {变量.shape}'
        elif 变量类型 == 'Tensor':
            显示内容 = f'{变量类型},矩阵, {变量.size()}'
        elif 变量类型 in ['list', 'tuple', 'set']:
            显示内容列表 = [以文本显示变量结构_深度缩进(子变量, 深度+1) for 子变量 in 变量]
            显示内容 = '\n'.join(显示内容列表)
            显示内容 = f':::{变量类型}\n{显示内容}'
        elif isinstance(变量, dict):
            显示内容列表 = [str(子变量名)+': '+以文本显示变量结构_深度缩进(子变量, 深度+1) for 子变量名,子变量 in 变量.items()]
            显示内容 = '\n'.join(显示内容列表)
            显示内容 = f':::{变量类型}\n{显示内容}'
        else:
            显示内容 = str(变量)
            if len(显示内容)>单元显示长度限制:
                显示内容 =  f'{显示内容[:单元显示长度限制]}...:::{变量类型}'
            else:
                显示内容 = f'{显示内容}:::{变量类型}'
    except Exception as e:
        print(e)
        显示内容 = f"...无法识别的类型{type(变量)}..."
    
    return f'{缩进}{显示内容}'


def 获取类名(frame):
    import inspect
    局部变量 = inspect.getargvalues(frame)[3]
    if 'self' in 局部变量:
        类 = getattr(局部变量['self'], '__class__')
        return getattr(类, '__name__')
    return None


def 查看变量结构(变量, 说明='' , 显示类=True, 显示方法=True):
    import inspect
    调用栈 = inspect.stack()
    方法名 = 调用栈[1][3]
    上下文 = str(调用栈[1][4][0]).strip()
    调用类名 = 获取类名(调用栈[1][0])
    显示内容 = f'{上下文}, 调用者: {调用类名}::{方法名}::::{说明}\n{以文本显示变量结构_深度缩进(变量, 深度=0)}\n'
    return 显示内容

def 显示变量结构(变量, 说明='', 显示类=True, 显示方法=True):
    import inspect
    调用栈 = inspect.stack()
    方法名 = 调用栈[1][3]
    上下文 = str(调用栈[1][4][0]).strip()
    调用类名 = 获取类名(调用栈[1][0])
    显示内容 = f'{上下文}, 调用者: {调用类名}::{方法名}::::{说明}\n{以文本显示变量结构_深度缩进(变量, 深度=0)}\n'
    print(显示内容)


def viewVar(var):
    return 以文本显示变量结构_深度缩进(var)
