def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]
    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc


def subclass_where(cls, **kwargs):
    k, v = next(iter(kwargs.items()))

    available_subcls = []
    for s in subclasses(cls):
        if hasattr(s, k):
            available_subcls.append(getattr(s, k))
            if getattr(s, k) == v:
                return s

    raise KeyError(
        f"No subclasses of {cls.__name__} with cls.{k} == '{v}'."
        f"Available subclasses: {available_subcls}"
    )
