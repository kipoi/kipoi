from kipoi.specs import RemoteFile
from kipoi.utils import recursive_dict_parse
import pytest


def test_recursive_dict_parse():
    a = {"a": 1,
         "b": 2}
    assert a == recursive_dict_parse(a, "url", RemoteFile.from_config)

    a = {"a": 1,
         "b": 2,
         "c": {"e": 3}}
    assert a == recursive_dict_parse(a, "url", RemoteFile.from_config)

    a = {"a": 1,
         "b": 2,
         "c": [{"url": "a",
                "md5": "b"}]}
    out = recursive_dict_parse(a, "url", RemoteFile.from_config)
    assert a != out
    assert out['a'] == a['a']
    assert out['b'] == a['b']
    assert out['c'] == [RemoteFile(url=a['c'][0]['url'],
                                   md5=a['c'][0]['md5'])]
