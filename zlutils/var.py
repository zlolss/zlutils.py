
class 字段类:
    def __init__(我, /, **预设的字段):
        for 字段名, 值 in 预设的字段.items():
            我.__setattr__(字段名, 值)
            