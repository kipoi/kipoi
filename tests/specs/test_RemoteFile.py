"""Test remote file
"""
from kipoi.specs import RemoteFile


def test_RemoteFile(tmpdir):
    f = RemoteFile("https://i12g-gagneurweb.in.tum.de/public/docs/concise/index.html", "a10361dc7f4582bb30eea089549bfc46")
    p = tmpdir.mkdir("sub").join("output")
    f.get_file(str(p))
    assert f.validate(str(p)) is False
