from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict

@dataclass
class KipoiModelDescription:
    args: Dict
    schema: Dict # Model schema class perhaps?
    defined_as: str 
    model_type: str = ""
    default_dataloader: str = '.'
    dependencies: Any # Dependencies class, a default value need to be added
    model_test: Any # Modeltest class, a default value need to be added
    path: str = "" 
    writers: Dict = OrderedDict()

