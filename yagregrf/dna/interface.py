from abc import ABC, abstractmethod


class GaussianRandomField(ABC):

    @abstractmethod
    def generate(self, nSamp):
        pass
