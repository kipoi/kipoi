from related.functions import to_dict


# HACK - custom convertion of tuple to string
@to_dict.register(tuple)
def _(obj, **kwargs):
    suppress_empty_values = kwargs.get("suppress_empty_values", False)
    # retain_collection_types = kwargs.get("retain_collection_types", False)

    if not suppress_empty_values or len(obj):
        return str(obj)
        # cf = obj.__class__ if retain_collection_types else list
        # return cf([to_dict(i, **kwargs) for i in obj])
