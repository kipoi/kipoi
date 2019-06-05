import os
import shutil
import cyvcf2
import numpy as np
from contextlib import contextmanager
from kipoi_utils.utils import _call_command

from kipoi.cli.env import *
from kipoi_conda.utils import *
import filelock

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




def create_model_env(model,source,tmpdir=None):
    if isinstance(model, str):
        model = [model]
    import uuid

    class Args(object):
        def __init__(self, model, source):
            self.model = model
            self.source = source        
        def _get_kwargs(self):
            kwargs = {"dataloader": [], "env":None, "gpu": False, "model": self.model, "source": self.source,
                  "tmpdir": "something", "vep": False}
            return kwargs

    # create the tmp dir
    if tmpdir is None:
        tmpdir = "/tmp/kipoi/envfiles/" + str(uuid.uuid4())[:8]
    else:
        tmpdir = args.tmpdir
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # write the env file
    logger.info("Writing environment file: {0}".format(tmpdir))


    env, env_file = export_env(model,
                               None,
                               source,
                               env_file=None,
                               env_dir=tmpdir,
                               env=None,
                               vep=False,
                               interpret=False,
                               gpu=False)

    args = Args(model=model, source=source)
    env_db_entry = generate_env_db_entry(args, args_env_overload=env)
    envdb = get_model_env_db()
    envdb.append(env_db_entry)
    envdb.save()

    # setup the conda env from file
    kipoi_conda.create_env_from_file(env_file)
    env_db_entry.successful = True

    # env is environment name
    env_db_entry.cli_path = get_kipoi_bin(env)
    get_model_env_db().save()





def create_env_if_not_exist(model,  source, bypass=False, use_filelock=True):


    import os
    lockfile = os.join(os.path.dirname(os.path.abspath(__file__)),'conda_create_env_filelock.lock')


    with lockfile:

        if not bypass:
            env_name = get_env_name(model,source=source)
        else:
            env_name = None

        # check if we already have that env
        if not env_exists(env_name):
            # create env and register hook to delete env later
            create_model_env(model=model, source=source)

            import atexit
            def call_atexit(env_name):
                try:
                    args = ["env","remove","-n",env_name]
                    _call_command('conda' ,extra_args=args)
                except:
                    pass
            atexit.register(call_atexit, env_name=env_name)

        return env_name