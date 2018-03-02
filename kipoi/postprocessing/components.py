import related
import enum
from kipoi.external.related.mixins import RelatedConfigMixin
from kipoi.external.related.fields import StrSequenceField, NestedMappingField, TupleIntField, AnyField, UNSPECIFIED


@enum.unique
class VarEffectFuncType(enum.Enum):
    logit = "logit"
    logit_ref = "logit_ref"
    logit_alt = "logit_alt"
    diff = "diff"
    deepsea_scr = "deepsea_scr"
    custom = "custom"


# @enum.unique
# class VarEffectRCTypes(enum.Enum):
#     seq_only = "seq_only"
#     none = "none"


@related.immutable(strict=True)
class VarEffectScoringFuncArgument(RelatedConfigMixin):
    # MAYBE - make this a general argument class
    doc = related.StringField("", required=False)
    name = related.StringField(required=False)
    type = related.StringField(default='str', required=False)
    optional = related.BooleanField(default=False, required=False)
    default = related.StringField(required=False)
    tags = StrSequenceField(str, default=[], required=False)  # TODO - restrict the tags


@related.immutable(strict=True)
class VarEffectScoringFunctions(RelatedConfigMixin):
    name = related.StringField(required=False, default="")
    type = related.ChildField(VarEffectFuncType, required=False)
    defined_as = related.StringField(required=False, default="")
    args = related.MappingField(VarEffectScoringFuncArgument, "name", required=False)
    default = related.BooleanField(required=False, default=False)


@related.immutable(strict=True)
class VarEffectDataLoaderArgs(RelatedConfigMixin):
    bed_input = related.SequenceField(str, required=False)
    scoring_functions = related.SequenceField(VarEffectScoringFunctions, default=[], required=False)


@related.immutable(strict=True)
class VarEffectModelArgs(RelatedConfigMixin):
    seq_input = related.SequenceField(str)
    use_rc = related.ChildField(bool, default=False, required=False)
    scoring_functions = related.SequenceField(VarEffectScoringFunctions,
                                              default=[], required=False)
