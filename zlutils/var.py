
def getDeepDict(dd: dict, str_path: str = "", default=None):
    # 从嵌套的字典中直接取值，避免报错
    # str_path sample: "llm_api.ollama.url"
    paths = str_path.split('.')
    cur = dd
    for path in paths:
        cur = cur.get(path, None)
        if cur is None:
            cur = default
            break
    return cur


class 字段类:
    def __init__(我, /, **预设的字段):
        for 字段名, 值 in 预设的字段.items():
            我.__setattr__(字段名, 值)

    def get(我, 字段名):
        return 我.__getattribute__(字段名)

    def __str__(我):
        return ','.join([f'{变量名}:{值}' for 变量名,值 in vars(我).items()])
            
    def __repr__(我):
        return 我.__str__()
