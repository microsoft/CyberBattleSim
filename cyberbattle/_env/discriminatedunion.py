# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A discriminated union space for Gym"""

from collections import OrderedDict
from typing import Mapping, Union, List

from gym import spaces
from gym.utils import seeding


class DiscriminatedUnion(spaces.Dict):  # type: ignore
    """
    A discriminated union of simpler spaces.

    Example usage:

    self.observation_space = discriminatedunion.DiscriminatedUnion(
        {"foo": spaces.Discrete(2), "Bar": spaces.Discrete(3)})

    """

    def __init__(self,
                 spaces: Union[None, List[spaces.Space], Mapping[str, spaces.Space]] = None,
                 **spaces_kwargs: spaces.Space) -> None:
        """Create a discriminated union space"""
        if spaces is None:
            super().__init__(spaces_kwargs)
        else:
            super().__init__(spaces=spaces)

    def seed(self, seed: Union[None, int] = None) -> None:
        self._np_random, seed = seeding.np_random(seed)
        super().seed(seed)

    def sample(self) -> object:
        space_count = len(self.spaces.items())
        index_k = self.np_random.randint(space_count)
        kth_key, kth_space = list(self.spaces.items())[index_k]
        return OrderedDict([(kth_key, kth_space.sample())])

    def contains(self, candidate: object) -> bool:
        if not isinstance(candidate, dict) or len(candidate) != 1:
            return False
        k, space = list(candidate)[0]
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
        return self.__class__.__name__ + "(" + ", ". join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n: object) -> object:
        return super().to_jsonable(sample_n)

    def from_jsonable(self, sample_n: object) -> object:
        ret = super().from_jsonable(sample_n)
        assert len(ret) == 1
        return ret

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiscriminatedUnion) and self.spaces == other.spaces


def test_sampling() -> None:
    """Simple sampling test"""
    union = DiscriminatedUnion(spaces={"foo": spaces.Discrete(8), "Bar": spaces.Discrete(3)})
    [union.sample() for i in range(100)]
