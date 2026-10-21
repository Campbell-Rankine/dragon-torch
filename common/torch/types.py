from typing import Union, Literal

type DeviceLike = Union[Literal['cpu'], Literal['cuda']]
type TorchFloatKind = Union[Literal['float'], Literal['float16'],
                            Literal['float32'], Literal['float64'], Literal['float128']]
type TorchIntegerKind = Union[Literal['integer'],
                              Literal['int64'], Literal['int32']]
