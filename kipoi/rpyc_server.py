import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
from rpyc.utils.server import ThreadedServer
import logging
import numpy
import os
from kipoi.model import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


"""
input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False)
 predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):


"""

class MyService(rpyc.Service):

    #######################################################
    # NOT FOR USERS
    def exposed__set_model_type(self, model_type):
        self.model_type = model_type
        if model_type != "custom":
            if model_type not in AVAILABLE_MODELS:
                raise RuntimeError("model must be in {}".format(str(AVAILABLE_MODELS.keys())))
            self.model_cls = AVAILABLE_MODELS[model_type]

    def exposed__initialize(self, *args, **kwargs):
        if self.model_type == "custom":
            self.model_cls = load_model_custom(*args, **kwargs)
             # it should inherit from Model
            assert issubclass(self.model_cls, BaseModel) 
            mod = Mod()
        else:
            self.model = self.model_cls(*args, **kwargs)
    #######################################################


    #######################################################
    # FOR USERS
    def exposed_predict_on_batch(self, x):
        x =  rpyc.classic.obtain(x)
        return self.model.predict_on_batch(x=x)

    def exposed_input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        return self.model.input_grad(x=x, filter_idx=filter_idx, avg_func=avg_func,
                   layer=layer, final_layer=final_layer,selected_fwd_node=selected_fwd_node, 
                   pre_nonlinearity=pre_nonlinearity)

    def exposed_predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        return self.model.predict_activation_on_batch(x=x, layer=layer, 
                   pre_nonlinearity=pre_nonlinearity)

    #######################################################
    # FOR GETTING THE RIGHT PATH
    def exposed_cd(self, newdir):
        os.chdir(os.path.expanduser(newdir))


    #######################################################
    # FOR DEBUG / TESTING
    def exposed_echo(self, x):
        return x



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('port', type=int, help='the port')


    parser.add_argument('--logging_level', type=int, help='debug level',
        default=10)


    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    logger.debug("using port %i",args.port)

    logger.debug("working dir {}".format(os.getcwd()))

    #raise RuntimeError(os.getcwd())


    server = ThreadedServer(MyService, port=args.port)
    server.start()