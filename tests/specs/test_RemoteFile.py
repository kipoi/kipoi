"""Test remote file
"""
from kipoi.specs import RemoteFile


def test_RemoteFile(tmpdir):
    f = RemoteFile("https://raw.githubusercontent.com/kipoi/kipoi/d35d30b10dd25cc5f6d85de48cb201f3c01d4b3e/README.md", "2ff2e3f12d0986fb7e32453f8451a3e2")
    p = tmpdir.mkdir("sub").join("output")
    f.get_file(str(p))
    assert f.validate(str(p)) is False
