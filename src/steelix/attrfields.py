from typing import Any, Dict, final

from ._type_inference import get_hint
from .attr import Attr, Ref
from .fields import Fields


class AttrFields(Fields[Attr]):
    """
    Fields subclass for storing attributes (Attr).
    The main difference in the behaviour is that attribute default values get implicitly wrapped in Attr.
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        base_kwargs = self.move_args_into_kwargs(*_args, **_kwargs)
        kwargs: Dict[str, Attr] = {}
        for name in base_kwargs:
            value = base_kwargs[name]
            value_type = get_hint(self.get_kwargs_types()[name])
            traceback_name = f"attribute {name}"
            if isinstance(value, Attr):
                assert value_type == value.value_type
                kwargs[name] = value
            elif isinstance(value, Ref):  # Function attribute references
                if value.value_type is not None:
                    assert value_type == value.value_type
                kwargs[name] = Attr(
                    value_type,
                    Ref(value_type, value.name, value.parent),
                    _traceback_name=traceback_name,
                )
            else:
                kwargs[name] = Attr(
                    value_type,
                    value,
                    _traceback_name=traceback_name,
                )
        super().__init__(**kwargs)


@final
class NoAttrs(AttrFields):
    pass
