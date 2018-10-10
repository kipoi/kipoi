import os
import shutil
import cyvcf2
import numpy as np


def compare_vcfs(fpath1, fpath2):
    fh1 = cyvcf2.VCF(fpath1)
    fh2 = cyvcf2.VCF(fpath2)
    for rec1, rec2 in zip(fh1, fh2):
        i1 = dict(rec1.INFO)
        i2 = dict(rec2.INFO)
        for k in i1:
            if ':rID' in k:
                continue
            min_round = min(len(i1[k]) - i1[k].index(".") - 1, len(i2[k]) - i2[k].index(".") - 1) - 2  # -2 for more tolerance
            assert np.round(float(i1[k]), min_round) == np.round(float(i2[k]), min_round)
    fh2.close()
    fh1.close()


def cp_tmpdir(example, tmpdir):
    from uuid import uuid4
    tdir = os.path.join(str(tmpdir), example, str(uuid4()))
    shutil.copytree(example, tdir)
    return tdir
