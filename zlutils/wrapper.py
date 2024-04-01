
class 篡改类:
    def __init__(我, 原始类):
        我.__原始类 = 原始类

    def __getattribute__(我, 方法名):
        try:
            return super().__getattribute__(方法名)
        except:
            try:
                return getattr(我.__原始类, 方法名)
            except AttributeError as e:
                raise AttributeError(e)
