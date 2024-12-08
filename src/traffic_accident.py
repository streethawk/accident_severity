import sys
sys.path.append("src/library")
sys.path.append("src/models")
from timeit import default_timer as timer
from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np

# import project libraries
from constants import *
from dataset import Dataset
from accident_model import AccidentModel
class TrafficAccident(object):
   def __init__(self):
        self._logger = f'::{self.__class__.__name__}::'
        self._dataset = None
        self._accident_model = None
        # self._classification = CLASSIFICATION
        # self.get_classification
        self._initialise()
        
   def get_classification(self):
        return self._classification

   def _initialise(self):
        start = timer()
        print(f'{self._logger}initialisation::...')
        self._load_dataset()
        #self._classification = self._dataset.get_unique_values('rating')
        #self._create_models()
        print(f'{self._logger}initialisation::Elapse time: {timedelta(seconds=timer()-start)}')
   
   def _load_dataset(self):
    print(f'{self._logger}loading UK Traffic Accidents dataset::...')
    self._dataset = Dataset(dataset_filename = DATASET_PATH + DATASET_FILENAME)
    
   def _create_models(self):
    # pass (mandatory) dataset 
    self._rating_model = AccidentModel(dataset = self._dataset)

   def explore_data(self):
        self._dataset.explore_data()
  

   def data_categorisation(self, level='all'):
        self._dataset.data_categorisation(level)
   
   def get_correlation(self, level='all'):
      self._dataset.get_corr(self, 0.80)


   def _print_classification(self):
        print(f'\n{COLOR_INVERSE} {self._dataset.get_next_step()}. Classification: {COLOR_RESET}')
        for i in range(len(self._classification)):
            print(f'{TAB_SPACE}{i+1:<5} {COLOR_BOLD}{COLOR_BLUE}{self._classification[i]}{COLOR_RESET}')
  
   def pie_chart(self):      
      pie_lst = ['Did_Police_Officer_Attend_Scene_of_Accident']
      for col in pie_lst:
        self._dataset.pie_chart(col)

   def keyfigures(self):
      cnt_lst1 = ['Road_Type', 'Junction_Control',
           'Pedestrian_Crossing-Human_Control',
           'Road_Surface_Conditions']

      for col in cnt_lst1:
        self._dataset.cnt_plot(col)

   def generate_plot(self):
    cnt_lst2 = ['Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
            'Weather_Conditions',
            'Special_Conditions_at_Site', 'Carriageway_Hazards']

    for col in cnt_lst2:
      self._dataset.generate_plot(col)

   def show_sample(self):
    self._dataset.show_sample()

   def get_stats(self):
    self._dataset.get_stats()

   def print_cat_cols(self):
     self._dataset.print_results()

