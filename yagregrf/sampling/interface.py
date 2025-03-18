from abc import ABC, abstractmethod


class SamplingEngine(ABC):

    @abstractmethod
    def generate_realisation(self):
        pass
