import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
from rpyc.utils.server import ThreadedServer
import logging
import numpy
import os

from kipoi_utils import (load_module, cd, merge_dicts, read_pickle, override_default_kwargs,
                    load_obj, inherits_from, infer_parent_class, makedir_exist_ok)

from kipoi.model import *

# try:
#     import concise
# except ImportError as e:
#     pass # module doesn't exist, deal with it


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def prepare_args(*args, **kwargs):
    args = [rpyc.classic.obtain(v) for v in args]
    kwargs = {k: rpyc.classic.obtain(v)  for k, v in kwargs.items()}
    return args, kwargs

def add_func_to_service(cls, funcs):
    if isinstance(funcs, str):
        funcs =  [funcs]
    # helper
    def make_func(name):
        def f(self, *args, **kwargs):
            args,kwargs = prepare_args( *args, **kwargs)
            return getattr(self.model, name)(*args, **kwargs)
        return f
        
    for name in funcs:
        setattr(cls, 'exposed_{}'.format(name),make_func(name))






class ModelRpycServiceBase(rpyc.Service):

    # #######################################################
    # # NOT FOR USERS
    # def exposed__set_model_type(self, model_type):
        
    #     self.model_type = model_type
    #     if model_type != "custom":
    #         if model_type not in AVAILABLE_MODELS:
    #             raise RuntimeError("model must be in {}".format(str(AVAILABLE_MODELS.keys())))
    #         self.model_cls = AVAILABLE_MODELS[model_type]

    def exposed__initialize(self, *args, **kwargs):
        if self.model_type == "custom":
            self.model_cls = load_model_custom(*args, **kwargs)
             # it should inherit from Model
            assert issubclass(self.model_cls, BaseModel) 
            self.model = Mod()
        else:
            self.model = self.model_cls(*args, **kwargs)

        self.pipeline = None

    def exposed_cd(self, newdir):

        try:
            os.chdir(newdir)
        except FileNotFoundError as e:
            raise FileNotFoundError("{} cwd {} newdir {}".format(str(e), os.getcwd(),newdir))
        #os.chdir(os.path.expanduser(newdir))

    def exposed_get_pipeline(self):
        return self.pipeline

    def exposed__init_pipeline(self, default_dataloader, source, model, cwd):
        default_dataloader = rpyc.classic.obtain(default_dataloader)
        source = rpyc.classic.obtain(source)  
        model = rpyc.classic.obtain(model)  
        cwd  = rpyc.classic.obtain(cwd )  


        assert isinstance(default_dataloader, str)

        with cd(cwd):



            # load from directory
            # attach the default dataloader already to the model
            if ":" in default_dataloader:
                dl_source, dl_path = default_dataloader.split(":")
            else:
                dl_source = source
                dl_path = default_dataloader

            # allow to use relative and absolute paths for referring to the dataloader
            default_dataloader_path = os.path.join("/" + model, dl_path)[1:]
            default_dataloader = kipoi.get_dataloader_factory(default_dataloader_path,
                                                              dl_source)

        self.pipeline = Pipeline(model=self.model, dataloader_cls=default_dataloader)


    def exposed__populate_model(self, **kwargs):
        _,kwargs = prepare_args(**kwargs)

        for k,v in kwargs.items():
            setattr(self.model, k, v)


    def exposed_echo(self, x):
        return x


add_func_to_service(ModelRpycServiceBase,[ "predict_on_batch","input_grad",
                                            "predict_activation_on_batch"])


class KerasModelRpycService(ModelRpycServiceBase):
    def __init__(self):
        super().__init__()
        self.model_type = "keras"
        self.model_cls = KerasModel

    def exposed_get_keras_backend(self):
        import keras
        return keras.backend._BACKEND


add_func_to_service(KerasModelRpycService, "get_layers_and_outputs")



class PyTorchModelRpycService(ModelRpycServiceBase):
    def __init__(self):
        super().__init__()
        self.model_type = "pytorch"
        self.model_cls = PytorchModel

class OldPyTorchModelRpycService(ModelRpycServiceBase):
    def __init__(self):
        super().__init__()
        self.model_type = "old_pytorch"
        self.model_cls = OldPyTorchModel

class SklearnModelRpycService(ModelRpycServiceBase):
    def __init__(self):
        super().__init__()
        self.model_type = "sklearn"
        self.model_cls = SklearnModel

class TensorflowModelRpycService(ModelRpycServiceBase):
    def __init__(self):
        super().__init__()
        self.model_type = "tensorflow"
        self.model_cls = TensorflowModel

class CustomModelRpycService(ModelRpycServiceBase):
    def __init__(self):
        super().__init__()
        self.model_type = "custom"
        self.model_cls = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('port', type=int,       help='the port')
    parser.add_argument('model_type', type=str, help='which model')

    parser.add_argument('--logging_level', type=int, help='debug level',
        default=0)


    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    logger.debug("using port %i",args.port)
    logger.debug("model_type %i",args.port)
    logger.debug("working dir {}".format(os.getcwd()))

    #raise RuntimeError(os.getcwd())
    model_type = args.model_type
    if  model_type == "keras":
        Service = KerasModelRpycService
    elif model_type == "pytorch":
        Service = PyTorchModelRpycService
    elif model_type == "old_pytorch":
        Service = OldPyTorchModelRpycService
    elif model_type == "sklearn":
        Service = SklearnModelRpycService
    elif model_type == "tensorflow":
        Service = TensorflowModelRpycService
    elif model_type == "custom":
        Service = CustomModelRpycService
    else:
        raise RuntimeError("'{}' is not a supported model type".format(model_type))


    server = ThreadedServer(Service, port=args.port, protocol_config=dict(allow_all_attrs=True, allow_public_attrs=True))
    server.start()