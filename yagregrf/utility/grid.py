from abc import ABC, abstractmethod


class Grid(ABC):

    def __init__(self, dim):

        self._dim = dim

    @property
    def dimension(self):
        return self._dim

    @abstractmethod
    @property
    def nVertices(self):
        pass

    @abstractmethod
    def coordinates(self):
        pass


class UniformGrid1d(Grid):
    pass


class UniformGrid2d(Grid):
    pass
