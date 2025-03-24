import logging

# set up logger
rfLogger = logging.getLogger(__name__)
rfLogger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)
rfLogger.addHandler(consoleHandler)


class RandomField:

    def __init__(self, engine, verbose=False):

        self._verbose = verbose
        self._engine = engine

    @property
    def engine(self):
        self._engine = engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine

    def generate(self, nSamples, nPrintIntervals=10):

        printInterval = nSamples // nPrintIntervals

        samples = []

        for n in range(nSamples):

            if self._verbose:

                if n == 0:
                    rfLogger.info(f"Start generating {nSamples} realisations")

                elif nSamples > nPrintIntervals and n % printInterval == 0:
                    rfLogger.info(f"{n} realisations generated")

            samples.append(self._engine.generate_realisation())

        return samples
