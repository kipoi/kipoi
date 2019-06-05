import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
rpyc.core.protocol.DEFAULT_CONFIG['propagate_SystemExit_locally'] = True




rpyc_connection_config = dict(allow_all_attrs=True, allow_public_attrs=True)
rpyc_protocol_config = dict(allow_all_attrs=True, allow_public_attrs=True)




def rpyc_classic_obtain(v):
    if rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle']:
        return rpyc.classic.obtain(v)
    else:
        return v

def prepare_args(*args, **kwargs):
    if rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle']:
        args = [rpyc.classic.obtain(v) for v in args]
        kwargs = {k: rpyc.classic.obtain(v)  for k, v in kwargs.items()}
        return args, kwargs
    else:
        return args,kwargs
