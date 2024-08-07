# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A discriminated union space for Gym"""

from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence, TypeVar, Union
from typing import Dict as TypingDict, Generic, cast
import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

T_cov = TypeVar("T_cov", covariant=True)


class DiscriminatedUnion(spaces.Dict, Generic[T_cov]):
    """
    A discriminated union of simpler spaces.

    Example usage:

    self.observation_space = discriminatedunion.DiscriminatedUnion(
        {"foo": spaces.Discrete(2), "Bar": spaces.Discrete(3)})

    Generic type T_cov is the type of the contained discriminated values.
    It should be defined as  a typed dictionary, e.g.:  TypedDict('Choices', {'foo': int, 'Bar': int})

    """

    def __init__(
        self,
        spaces: Union[None, TypingDict[str, spaces.Space]] = None,
        seed: Optional[Union[dict, int, np.random.Generator]] = None,
        **spaces_kwargs: spaces.Space,
    ) -> None:
        """Create a discriminated union space"""
        if spaces is None:
            super().__init__(spaces_kwargs)
        else:
            super().__init__(spaces=spaces, seed=seed)

        if isinstance(seed, dict):
            self.union_np_random, _ = seeding.np_random(None)
        elif isinstance(seed, np.random.Generator):
            self.union_np_random = seed
        else:
            self.union_np_random, _ = seeding.np_random(seed)

    def seed(self, seed: Union[dict, None, int] = None):
        return super().seed(seed)

    def sample(self, mask=None) -> T_cov:  # type: ignore
        space_count = len(self.spaces.items())
        index_k = self.union_np_random.integers(0, space_count)
        kth_key, kth_space = list(self.spaces.items())[index_k]
        return cast(T_cov, OrderedDict([(kth_key, kth_space.sample())]))

    def contains(self, x) -> bool:
        if not isinstance(x, dict) or len(x) != 1:
            return False
        k, space = list(x)[0]
        return k in self.spaces.keys()

    @classmethod
    def is_of_kind(cls, key: str, sample_n: Mapping[str, object]) -> bool:
        """Returns true if a given sample is of the specified discriminated kind"""
        return key in sample_n.keys()

    @classmethod
    def kind(cls, sample_n: Mapping[str, object]) -> str:
        """Returns the discriminated kind of a given sample"""
        keys = sample_n.keys()
        assert len(keys) == 1
        return list(keys)[0]

    def __getitem__(self, key: str) -> spaces.Space:
        return self.spaces[key]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
        return super().to_jsonable(sample_n)

    def from_jsonable(self, sample_n: TypingDict[str, list]) ->  list[OrderedDict[str, Any]]:
        ret = super().from_jsonable(sample_n)
        assert len(ret) == 1
        return ret

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiscriminatedUnion) and self.spaces == other.spaces


def test_sampling() -> None:
    """Simple sampling test"""
    union = DiscriminatedUnion(spaces={"foo": spaces.Discrete(8), "Bar": spaces.Discrete(3)})
    [union.sample() for i in range(100)]
