class AccidentModel(object):
   def __init__(self):
    self._logger = f'::{self.__class__.__name__}::'
    self._initialise()

    def _initialise(self):
        start = timer()
        print(f'{self._logger}initialisation::...')
        #self._load_dataset()
        #self._classification = self._dataset.get_unique_values('rating')
        #self._create_models()
        print(f'{self._logger}initialisation::Elapse time: {timedelta(seconds=timer()-start)}')
        