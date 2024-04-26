"""Test remote file
"""
from kipoi.specs import RemoteFile


def test_RemoteFile(tmpdir):
    remote_url = "https://raw.githubusercontent.com/kipoi/kipoi/d35d30b10dd25cc5f6d85de48cb201f3c01d4b3e/README.md"
    remote_mdf5sum = "2ff2e3f12d0986fb7e32453f8451a3e2"
    
    f = RemoteFile(remote_url, remote_mdf5sum)
    p = tmpdir.mkdir("sub").join("output")
    f.get_file(str(p))
    assert f.validate(str(p)) is True
    
    f = RemoteFile(remote_url, "some_wrong_md5_hash")
    assert f.validate(str(p)) is False
