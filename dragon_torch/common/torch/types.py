from typing import Union, Literal

type RLTypes = Union[Literal['torch-tensor'], Literal['numpy-array']]
type DeviceLike = Union[Literal['cpu'], Literal['cuda']]
type TorchFloatKind = Union[Literal['float'], Literal['float16'],
                            Literal['float32'], Literal['float64']]
type TorchIntegerKind = Union[Literal['int'], Literal['int8'], Literal['int16'],
                              Literal['int64'], Literal['int32']]
