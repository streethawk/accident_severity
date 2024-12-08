# Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from timeit import default_timer as timer

# Data Visualization Libraries
import plotly.express as px
from matplotlib.patches import ConnectionPatch

# Statistical Libraries
import scipy.stats as stats

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)

# Oversampling Techniques for Imbalanced Data
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error

# import project libraries
from constants import *
# from utility import *

class Dataset(object):
   def __init__(self, dataset_filename = None):
        self._logger = f'::{self.__class__.__name__}::'
        self._dataset_filename = dataset_filename
        self._dataset = None
        self._X_resampled = None
        self._y_resampled = None
        # self._output_column = DATASET_COLUMN_RATING
        self._process_step = 0
        self.X_inputs = None
        self.y_output = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._data_split = None
        self.cat_cols = None
        self.num_cols = None
        self._initialise()

   def _initialise(self):
        start = timer()
        print(f'{self._logger}initialisation::...')
        if self._dataset_filename is not None:
            self._load_dataset()
        print(f'{self._logger}initialisation::Elapse time: {timedelta(seconds=timer()-start)}')
   
   def _load_dataset(self):
        self._dataset = pd.read_csv(self._dataset_filename)
   
   def explore_data(self):
        self.reset_process_step()
        self._print_dataset_size()
        self._print_dataset_columns()
        
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Drop Columns: {COLOR_RESET}')
        self._dataset.drop(columns=['Unnamed: 0', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 
                 'Local_Authority_(Highway)', 'LSOA_of_Accident_Location'], inplace=True)

        self._print_dataset_head(nrow = 5, ncol = 3)
        self._print_dataset_null_count()
        
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Dataset Info: {COLOR_RESET}')
        print(f'\n {self._dataset.info()}')

        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Dataset Null: {COLOR_RESET}')
        print(f'\n {self._dataset.isna().any()}')

        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Dataset % Null: {COLOR_RESET}')
        print(f'\n {self._dataset.isnull().sum() / len(self._dataset) * 100}')

        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Drop Rows: {COLOR_RESET}')
        self._dataset.dropna(subset=['Longitude', 'Time', 'Pedestrian_Crossing-Human_Control', 
                  'Pedestrian_Crossing-Physical_Facilities'], inplace=True)

        dup_rows = self._dataset[self._dataset.duplicated()]
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Number of Duplicated Rows: {COLOR_RESET}')
        print(f'\n {dup_rows.shape[0]}')

        self._dataset.drop_duplicates(inplace=True)
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Number of Rows remaining: {COLOR_RESET}')
        print(f'\n {self._dataset.shape[0]} ')

        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Describing the shape of remaining Data: {COLOR_RESET}')
        print(f'\n {self._dataset.describe(include=np.number) }')
        print(f'\n {self._dataset.describe(include=object) }')
      


   def reset_process_step(self):
        self._process_step = 0

   def get_next_step(self):
        self._process_step += 1
        return self._process_step

   def _print_dataset_size(self):
      print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Dataset: {COLOR_RESET}')
      label = 'Filename:'
      print(f'{TAB_SPACE}{label:<15} {COLOR_BLUE}{self._dataset_filename}{COLOR_RESET}')
      label = 'Total Rows:'
      print(f'{TAB_SPACE}{label:<15} {COLOR_BLUE}{self._dataset.shape[0]:,}{COLOR_RESET}')
      label = 'Total Columns:'
      print(f'{TAB_SPACE}{label:<15} {COLOR_BLUE}{self._dataset.shape[1]}{COLOR_RESET}')

   def _print_dataset_columns(self):
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Column definition: {COLOR_RESET}')
        for index, column in enumerate(self._dataset.columns):
            datatype_color = self._get_datatype_color(self._dataset[column].dtypes)
            print(f'{TAB_SPACE}{index+1:<5} {COLOR_BLUE}{column:<35}{COLOR_RESET} {COLOR_BOLD}{datatype_color}{self._dataset[column].dtypes}{COLOR_RESET}')

   def _print_dataset_head(self, nrow = 6, ncol = 4):
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Head: first {nrow} rows {COLOR_RESET}')
        # self.print_format_dataset(self._dataset, nrow, ncol)
        print(self._dataset.sample(5))

   def _print_dataset_null_count(self):
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Column Null Counts: {COLOR_RESET}')
        index = 0
        null_counts = self._dataset.isnull().sum()
        for column, value in null_counts.items():
            index += 1
            print(f'{TAB_SPACE}{index:<5} {COLOR_BLUE}{column:<35}{COLOR_RESET} {COLOR_BOLD}{COLOR_GREEN}{value}{COLOR_RESET}')

   def _get_datatype_color(self, datatype):
      color = COLOR_RESET
      if datatype == 'float64':
        color = COLOR_GREEN
      elif datatype == 'object':
        color = COLOR_PURPLE
      else:
        color = COLOR_BLACK
      return color
    
   def print_format_dataset(self, dataset, nrow, ncol, **kwargs):
      def _column_attribute(column, key, **kwargs):
          value = None
          if ('columns' in kwargs and
              column in kwargs['columns'] and
              key in kwargs['columns'][column]):
              value = kwargs['columns'][column][key]
          return value

   def data_categorisation(self, level='all'):
        self.reset_process_step()
        numerical_data = self._dataset.select_dtypes(include='number')
        self.num_cols = numerical_data.columns
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Count of Numerical Columns')
        print(f'\n {len(self.num_cols)}')
        print(f'\n {(self.num_cols)}')

        categorical_data = self._dataset.select_dtypes(include='object')

        self.cat_cols = categorical_data.columns
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Count of Categorical Columns')
        print(f'\n {len(self.cat_cols)}')
        print(f'\n {(self.cat_cols)}')

        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Box Plot')
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(20, 50))
        fig.subplots_adjust(right=1.5)

        for plot in range(1, len(self.num_cols)+1):
            plt.subplot(6, 4, plot)
            sns.boxplot(y=self._dataset[self.num_cols[plot-1]])

        plt.show()

        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Generate Diagnostic Plot')
        dist_lst = ['Police_Force', 'Accident_Severity',
                  'Number_of_Vehicles', 'Number_of_Casualties', 
                  'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number',
                  'Speed_limit', '2nd_Road_Class', '2nd_Road_Number',
                  'Urban_or_Rural_Area']

        for col in dist_lst:
          self.diagnostic_plot(self._dataset, col)

        
   def diagnostic_plot(self, data, col):
        fig = plt.figure(figsize=(20, 5))
        fig.subplots_adjust(right=1.5)
        
        plt.subplot(1, 3, 1)
        sns.distplot(data[col], kde=True, color='teal')
        plt.title('Histogram')
        
        plt.subplot(1, 3, 2)
        stats.probplot(data[col], dist='norm', fit=True, plot=plt)
        plt.title('Q-Q Plot')
        
        plt.subplot(1, 3, 3)
        sns.boxplot(data[col],color='teal')
        plt.title('Box Plot')
        
        plt.show()

   def get_correlation_matrix(self):
        print(f'\n{COLOR_INVERSE} {self.get_next_step()}. Generate Correlation Matrix')
        plt.figure(figsize = (15,10))
        numerical_data = self._dataset.select_dtypes(include=np.number)
        corr = numerical_data.corr(method='spearman')
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
        cormat.set_title('Correlation Matrix')
        plt.show()

   def get_corr(self, data, threshold):

    self.get_correlation_matrix()
    self._dataset.drop(columns=['Local_Authority_(District)'], 
        axis=1, inplace=True)    

   def pie_chart(self, col):      
      x = self._dataset[col].value_counts().values
      plt.figure(figsize=(7, 6))
      plt.pie(x, center=(0, 0), radius=1.5, labels=self._dataset[col].unique(), 
              autopct='%1.1f%%', pctdistance=0.5)
      plt.axis('equal')
      plt.show() 

   def cnt_plot(self, col):
       plt.figure(figsize=(15, 7))
       ax1 = sns.countplot(x=col, data=self._dataset,palette='rainbow')

       for p in ax1.patches:
         ax1.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1), ha='center')

       plt.show()

       print('\n')

   def generate_plot(self, col):
    plt.figure(figsize=(10, 7))
    sns.countplot(y=col, data=self._dataset,palette='rainbow')
    plt.show()

    print('\n')    

   def show_sample(self):
      print(self._dataset.sample(5))         

   def get_stats(self):
      print('Urban Or Rural Area count {self._dataset[\'Urban_or_Rural_Area\'].value_counts()}')
      print(self._dataset['Urban_or_Rural_Area'].replace(3, 1, inplace=True))
      self._dataset['Accident_Severity'].value_counts()
      self._dataset['Number_of_Vehicles'].value_counts()[:10]
      self._dataset['Number_of_Casualties'].value_counts()[:10]

      dt1 = self._dataset.groupby('Date')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})
      fig = px.line(dt1, x='Date', y='No. of Accidents',
              labels={'index': 'Date', 'value': 'No. of Accidents'})
      fig.show() 

      dt2 = self._dataset.groupby('Year')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})
      fig = px.line(dt2, x='Year', y='No. of Accidents',
              labels={'index': 'Year', 'value': 'No. of Accidents'})
      fig.show() 

      dt3 = self._dataset.groupby('Day_of_Week')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})
      fig = px.line(dt3, x='Day_of_Week', y='No. of Accidents',
              labels={'index': 'Day_of_Week', 'value': 'No. of Accidents'})
      fig.show()

   def print_results(self):
      print(self.cat_cols)
      len(self._dataset['Accident_Index'].unique())
      self._dataset.drop('Accident_Index',axis=1,inplace=True)


      plt.figure(figsize=(18, 5))
      sns.heatmap(self._dataset.isna(), yticklabels=False, cbar=False, cmap='viridis')
      plt.show()


      X = self._dataset[self._dataset.select_dtypes(include='number').columns]
      X = X.drop(columns=['Accident_Severity'], axis=1)
      plt.figure(figsize=(8, 10))
      X.corrwith(self._dataset['Accident_Severity']).plot(kind='barh',
                                      title="Correlation with 'Accident_Severity' column -")
      plt.show()


      cat_cols=[feature for feature in self._dataset.columns if self._dataset[feature].dtype=='O']
      print(cat_cols)


      for feature in cat_cols:
         print(f'The {feature} has following number of {len(self._dataset[feature].unique())}')
     
      labelencoder=LabelEncoder()
      for feature in cat_cols:
         self._dataset[feature]=labelencoder.fit_transform(self._dataset[feature])


      self._dataset.drop('Year',axis=1,inplace=True)
     
      self._dataset['Date'] = pd.to_datetime(self._dataset['Date'])
      self._dataset["day"] = self._dataset['Date'].dt.day
      self._dataset["month"] = self._dataset['Date'].dt.month
      self._dataset["year"] = self._dataset['Date'].dt.year


      self._dataset.drop("Date",axis=1,inplace=True)
      self._dataset.drop("Time",axis=1,inplace=True)


      self._dataset['Accident_Severity']=self._dataset['Accident_Severity'].map({1:0,2:1,3:2})


      dfnew=self._dataset[['Latitude','Longitude','day','month','1st_Road_Number','year','Day_of_Week','Accident_Severity']]


      from sklearn.preprocessing import StandardScaler
      features = [feature for feature in dfnew.columns if feature!='Accident_Severity']
      x = dfnew.iloc[0:50000, :-1]
      y = dfnew.iloc[0:50000,[-1]]
      x = StandardScaler().fit_transform(x)




      from imblearn.over_sampling import RandomOverSampler
      from sklearn.model_selection import train_test_split
      from imblearn.over_sampling import SMOTE


      oversample = RandomOverSampler()
      x, y = oversample.fit_resample(x, y)


      X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


      # Models to evaluate
      models = {
        "Extra Trees": ExtraTreesClassifier(random_state=0),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
     }

    # Step 8: Evaluate models
      for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        # Print metrics
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")

        # Classification Report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, preds))

        # Confusion Matrix
        conmat = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conmat, annot=True, cbar=False, cmap="Blues", fmt="d")
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

