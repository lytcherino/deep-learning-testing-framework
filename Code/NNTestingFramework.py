#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########
# ANN
###########
import matplotlib
#matplotlib.use('Agg') # Don't display plots in console
import matplotlib.pyplot as plt

import numpy as np

from DataHandler import DataHandler

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import callbacks
from keras import backend as K

import itertools, datetime, os, re

from sklearn.model_selection import KFold

from collections import defaultdict

seed = 1
np.random.seed(seed)

class NDimArray(object):
    
    def __init__(self):
        self.ndict = defaultdict(dict)

    def insert(self, indicies, value):
        """
        
        Insert 'value' at the indices specified by 'indices'
        
        args:
            (list)   array
            (any)    value
        
        """
        self.ndict = self.insertHelper(self.ndict, indicies, value)
        
    def insertHelper(self, dict_, array, value, index = 0):
      while index + 1 < len(array):
        if isinstance(dict_[array[index]], defaultdict):
          dict_[array[index]] = self.insertHelper(dict_[array[index]], array, value, index + 1)
        else:
          dict_[array[index]] = self.insertHelper(defaultdict(dict), array, value, index + 1)
        return dict_
      dict_[array[index]] = value 
      return dict_
        
    def get(self, indicies):
        """ 
        Use a list of indicies to index the structure
        as opposed to using the typical approach of brackets: [][][]...
        which is not convinent at run time
        
        Returns empty dict ({}) if there is no value at index
        """
        return self.getHelper(self.ndict, indicies)
    
    def getHelper(self, dict_, indicies, index = 0):
        while index + 1 < len(indicies):
            return self.getHelper(dict_[indicies[index]], indicies, index + 1)
        return dict_[indicies[index]]
        

class ModelBuilder:
    
    @staticmethod
    def hidden_dropout(features, parameters, optimizer = 'adam', init='glorot_uniform', activation = 'relu'):
        """
        parameteres[0]      - # Hidden Units
        parameters[1]       - Dropout Rate
        """
        #print(__name__,':',parameters)
        
        model = Sequential()
        
        # Input layer and first hidden layer
        model.add(Dense(units = parameters[0], kernel_initializer = init, input_dim = features, activation=activation))
        model.add(Dropout(parameters[1]))
    
        #model.add(Dropout(dropout_rate))
    
        model.add(Dense(output_dim = 1, activation='linear'))
    
        model.compile(optimizer = optimizer, 
                      loss = 'mean_squared_error',
                      metrics = ['mse'])
        return model
    
    @staticmethod
    def hidden_layers(features, parameters, optimizer = 'adam', init='glorot_uniform', activation = 'relu', dropout_rate = 0.0):
        """
        parameters[0]       - # Hidden Units
        parameters[1]       - Dropout Rate
        parameters[2]       - # Hidden Layers
        """

        model = Sequential()
        
        # Input layer and first hidden layer
        model.add(Dense(units = parameters[0], kernel_initializer = init, input_dim = features, activation=activation))
        model.add(Dropout(parameters[1]))
        
        for i in range(parameters[2]):
            model.add(Dense(units = parameters[0], kernel_initializer = init, activation=activation))
            model.add(Dropout(parameters[1]))

        model.add(Dense(output_dim = 1, activation='linear'))
    
        model.compile(optimizer = optimizer, 
                      loss = 'mean_squared_error',
                      metrics = ['mse'])
        return model
    
    @staticmethod
    def hidden_layers_drop(features, parameters, optimizer = 'adam', init='glorot_uniform', activation = 'relu', dropout_rate = 0.0):
        """
        parameters[0]       - # Hidden Units
        parameters[1]       - Dropout Rate
        parameters[2]       - # Hidden Layers
        parameters[3]       - Hidden Layer Drop Rate
        parameters[4]       - Hidden Units in Additional Layer
        """

        model = Sequential()
        
        # Input layer and first hidden layer
        model.add(Dense(units = int(parameters[0]), kernel_initializer = init, input_dim = features, activation=activation))
        model.add(Dropout(parameters[1]))
        
        if parameters[2] == 1:
            model.add(Dense(units = int(parameters[4]), kernel_initializer = init, activation=activation))
            model.add(Dropout(parameters[3]))
            
        elif parameters[2] == 2:
            model.add(Dense(units = int(parameters[4]), kernel_initializer = init, activation=activation))
            model.add(Dropout(parameters[3]))
            model.add(Dense(units = int(parameters[4]), kernel_initializer = init, activation=activation))
            model.add(Dropout(parameters[3]))

        model.add(Dense(output_dim = 1, activation='linear'))
    
        model.compile(optimizer = optimizer, 
                      loss = 'mean_squared_error',
                      metrics = ['mse'])
        return model
    
    @staticmethod
    def hidden_layers_drop_same(features, parameters, optimizer = 'adam', init='glorot_uniform', activation = 'relu', dropout_rate = 0.0):
        """
        parameters[0]       - # Hidden Units
        parameters[1]       - Dropout Rate
        parameters[2]       - # Hidden Layers
        """

        model = Sequential()
        
        # Input layer and first hidden layer
        model.add(Dense(units = int(parameters[0]), kernel_initializer = init, input_dim = features, activation=activation))
        model.add(Dropout(parameters[1]))
        
        for i in range(parameters[2]):
            model.add(Dense(units = int(parameters[0]), kernel_initializer = init, activation=activation))
            model.add(Dropout(parameters[1]))

        model.add(Dense(output_dim = 1, activation='linear'))
    
        model.compile(optimizer = optimizer, 
                      loss = 'mean_squared_error',
                      metrics = ['mse'])
        return model

        
class StatusTracker(object):
    
    def __init__(self, dependent_parameter, benchmark_parameter, verbose = 0, filename = None):
        self.epoch = 0
        self.dependent_parameter = dependent_parameter
        self.benchmark_parameter = benchmark_parameter
        self.benchmark_parameter_value = None
        self.verbose = verbose
        self.filename = filename
        self.epochs_without_better_value = 0
        self.writeToFile('Output for training started at ' + f"{datetime.datetime.now():%Y-%m-%d-%H-%-M-%S}" + ' for ' + self.prettyPrint() + '\n')
        
    def update(self, epoch, logs={}):
        if self.verbose >= 5:
            print('logs:',logs)
        
        if self.benchmark_parameter_value is None or logs[self.benchmark_parameter] < self.benchmark_parameter_value:
            self.benchmark_parameter_value = logs[self.benchmark_parameter]
            self.epoch = epoch
            self.epochs_without_better_value = 0
            
            if self.verbose >= 1:
                print(self.status())
            if self.verbose >= 3:
                self.writeToFile(self.status()+'\n')
        else:
            self.epochs_without_better_value += 1
            if self.verbose >= 2:
                print('Epochs without better {}: {}'.format(self.benchmark_parameter, self.epochs_without_better_value))
        
    def prettyPrint(self):
        string = ''
        for key, value in self.dependent_parameter.items():
            string += str(key) + ' = ' + str(value)
            string += ' | '
        return string
    
    def writeToFile(self, content):
        if self.filename is not None:
            with open(self.filename, 'a') as f:
                f.write(content)
    
    def status(self):
        return '(ST) - {}: Lowest value of {} at epoch {}: {}.'.format(self.prettyPrint(), self.benchmark_parameter, self.epoch, self.benchmark_parameter_value)
            
    def end(self):
        if self.verbose > 0:
            print(self.status())
        self.writeToFile('\n\nTraining ended - ' + self.status() + '\n\n')

class CustomHistory(callbacks.Callback):
    
    def __init__(self, dependent_variables, benchmark_parameter, verbose = 0, filename = None):
        self.verbose = verbose
        self.statusTracker = StatusTracker(dependent_variables, benchmark_parameter, verbose, filename)
        self.dependent_variables = dependent_variables
        
    def on_epoch_end(self, epoch, logs={}):
        self.statusTracker.update(epoch, logs)
        
    def on_train_end(self, logs={}):
        self.statusTracker.end()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return  

    
    
    
class TestingFramework:
    
    def __init__(self, modelBuilder, baseDirectory = './ANN/Output'):
                
        self.modelFilePath = None
        self.model_directory_name = None
        self.log_directory_name = None
        self.output_directory_name = None
        self.model_ind_directory_name = None
        
        self.modelList = []
        self.model_parameter_delimiter = '-'
        self.model_evaluation = dict()
        self.model_data = NDimArray()
        self.base_model_name = 'best_model'
        self.base_model_ext = '.hdf5'
        self.model_directory_name = 'Models/'
        self.final_model_directory_name = 'Final/'
        self.evaluation_raking_filename = 'ranked_evaluation.txt'
        
        self.base_log_name = 'log'
        self.base_log_ext = '.txt'
        self.log_directory_name = 'Logs/'
        
        self.evaluation_directory_name = 'Evaluation/'

        
        self.model_parameter_name = NDimArray()
        
        # Used to track the ranges for the argument to the network
        self.iterators = None
        # Names of the parameters
        self.parameters = None
        
        # Function used to construct the network
        self.modelBuilder = modelBuilder

        
        # Create directories to save files in
        self.createDirectories(baseDirectory)
        
    def reset(self):
        self.iterators = None
        self.parameters = None
        self.model_evaluation = dict()
        self.modelList = []
        self.model_data = NDimArray()
        self.model_parameter_name = NDimArray()
        
    def createDirectories(self, baseDirectory = './ANN/Output'):
        # Prepare directory structure
        currentTime = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"     
        self.output_directory_name = '{}/{}/'.format(baseDirectory, currentTime)
        os.makedirs(os.path.dirname(self.output_directory_name), exist_ok=True)
        
        self.model_directory_name = self.output_directory_name + self.model_directory_name 
        self.log_directory_name = self.output_directory_name + self.log_directory_name
        self.evaluation_directory_name = self.output_directory_name + self.evaluation_directory_name
        
        # Create model, log, evaluation directory
        os.makedirs(os.path.dirname(self.model_directory_name), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_directory_name), exist_ok=True) 
        os.makedirs(os.path.dirname(self.evaluation_directory_name), exist_ok=True) 

    def extract_it_param(self, iterators):
      iterators_ = []
      parameters_ = []
      
      for key in iterators:
        for keys_, values in iterators[key].items():
          parameters_.append(keys_)
          iterators_.append(values)
      return iterators_, parameters_

    def train(self, X_train, y_train, finalModel = False,
              X_val = None, y_val = None,
              max_epochs = 100000000, batch_size = 8, patience = 100, monitor = 'val_loss',
              **iterators):
        
        #print('iterators:',iterators)
        self.iterators, self.parameters = self.extract_it_param(iterators)
        print('extracted_iterators:',self.iterators)
        print('extracted_parameters:',self.parameters)

        assert(len(self.iterators) == len(self.parameters)), "Number of iterators and parameters must match"
        
        earlyStopping = EarlyStopping(monitor=monitor,
                                      min_delta=0,
                                      patience=patience,
                                      verbose=0, mode='auto')
        
        for item in itertools.product(*self.iterators):
                        
            features = X_train.shape[1]
            #print('ITEM:',item)
            
            self.model = self.modelBuilder(features, list(item))
            
            # Create model parameter name
            model_name = ''
            for i in range(len(item)):
                model_name += str(item[i])
                if i + 1 < len(item):
                    model_name += self.model_parameter_delimiter
            model_name += '/'
            
            if finalModel:
                self.model_ind_directory_name = self.model_directory_name + self.final_model_directory_name
                os.makedirs(os.path.dirname(self.model_ind_directory_name), exist_ok=True)
                self.modelFilePath = self.model_ind_directory_name + self.base_model_name + self.base_model_ext
                self.modelList.append(self.modelFilePath)
            else:
                self.model_ind_directory_name = self.model_directory_name + model_name
                os.makedirs(os.path.dirname(self.model_ind_directory_name), exist_ok=True)
                self.modelFilePath = self.model_ind_directory_name + self.base_model_name + self.base_model_ext
                self.modelList.append(self.modelFilePath)
            
            modelCheckPoint = ModelCheckpoint(self.modelFilePath, 
                                              monitor=monitor, 
                                              verbose=0, 
                                              save_best_only=True, 
                                              save_weights_only=False, 
                                              mode='auto', 
                                              period=1)
            
            # Create a map of parameter name : current index
            # e.g. {"Hidden Layer" : 1, "Units" : 3}
            parameter_index_dict = dict()
            for i in range(len(item)):
                parameter_index_dict[self.parameters[i]] = item[i]
            
            # Create log parameter name
            log_name = self.base_log_name
            for i in range(len(item)):
                log_name += '-' + str(item[i])
            log_name += self.base_log_ext
            
            customHistory = CustomHistory(parameter_index_dict, 
                                           'val_loss',
                                           verbose = 0, 
                                           filename = self.log_directory_name + log_name)
            
            callback_list = [modelCheckPoint, earlyStopping, customHistory]
            
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
                
            model_fit = self.model.fit(X_train, y_train, 
                                       validation_data = validation_data, 
                                       epochs = max_epochs, 
                                       batch_size = batch_size, 
                                       verbose = 0,
                                       callbacks = callback_list)
            
            self.model_data.insert(item, model_fit.history)
            self.model_parameter_name.insert(item, parameter_index_dict)
            
            if finalModel:
                return model_fit.history
            
            
    def plotAll(self, best_epoch_lines_to_draw = 10, width_height_ratio = [4,4], scale = 0.25):
        
        # Get all pairs of unique combinations of the parameters
        # and produce plots for each combination
        for parameter_names in itertools.combinations(self.parameters, 2):
            # Get index of the parameter to find the correct iterator
            index_of_parameter = []
            index_of_non_parameter = []
            for i in range(len(parameter_names)):
                for j in range(len(self.parameters)):
                    if parameter_names[i] == self.parameters[j]:
                        index_of_parameter.append(j)
                    else:
                        index_of_non_parameter.append(j)
                        
            #print('Index_of_parameter:',index_of_parameter)
            
            # Given index of the parameters used for the particular combination
            # get the 2 iterators
            
            # However given that self.iterators has N iterators, there are N - 2 iterators
            # that are not considered. These iterators are kept constant for each plot
            # Meaning that every plot for the 2 iterators selected has to be plotted 
            # for every possible value of the remaining iterators
            
            iterators = []
            const_iterators = []
            for i in range(len(self.iterators)):
                if i in index_of_parameter:
                    iterators.append(self.iterators[i])
                else:
                    const_iterators.append(self.iterators[i])
                    
                    
            #print('iterators: ',iterators, ' const_iterators: ', const_iterators)
                 
            max_size_iterators = []
            min_size_iterators = []
            for iterator in iterators:
                max_size_iterators.append(iterator[-1])
                min_size_iterators.append(iterator[0])
            
            #print('Min_size_iterators: ',min_size_iterators, 'Max_size_iterators: ', max_size_iterators)
            
            min_iterators = []
            for i in range(len(min_size_iterators)):
                    min_iterators.append([iterators[i][0]])
            
            #print('Const_iterators: ',const_iterators, 'Min_iterators: ', min_iterators)
           
            # Generate indices and plot
            for value in itertools.product(*const_iterators, *min_iterators):
                indices = np.zeros(len(self.parameters))
                const_count = 0
                count = len(const_iterators)
                for i in range(len(indices)):
                    if i in index_of_parameter:
                        indices[i] = value[count]
                        count += 1
                    else:
                        indices[i] = value[const_count]
                        const_count += 1
                
                # Plot for all cases where the const_iterators are kept constant
                #print('Indicies (1): ', list(indices))
                self.plot(list(indices), index_of_parameter, width_height_ratio, scale)
            
    def plot(self, indices, index_of_parameter, width_height_ratio, scale):

        length_iterator_1 = len(self.iterators[index_of_parameter[0]])
        length_iterator_2 = len(self.iterators[index_of_parameter[1]])
        
        #print('length_iterator_1:', length_iterator_1, 'length_iterator_2:', length_iterator_2)
        
        fig, ax_array = plt.subplots(length_iterator_1, length_iterator_2, 
                                     figsize=(int(length_iterator_1 + (length_iterator_2/length_iterator_1)*width_height_ratio[0]*scale),
                                              int(length_iterator_2+(length_iterator_1/length_iterator_2)*width_height_ratio[1]*scale)),
                                              squeeze = False)
        
        # List of parameters that are varying; not kept constant in the plot
        parameter_list = [parameter for index in index_of_parameter for parameter in self.parameters if self.parameters[index] == parameter]
                            
        for i, ax_row in enumerate(ax_array):
            for j, axes in enumerate(ax_row):

                # Update indices
                indices[index_of_parameter[0]] = self.iterators[index_of_parameter[0]][i-1]
                indices[index_of_parameter[1]] = self.iterators[index_of_parameter[1]][j-1]

                # Determines how many vertical lines are drawn in the graph
                best_epoch_lines_to_draw = 10;
                
                # Construct parameter title
                parameters_non_constant = ''
                parameters_constant = ''
                
                #print('Indicies (2):', indices)
                parameter_name_dict = self.model_parameter_name.get(indices)
                for key, value in parameter_name_dict.items():
                    if key in parameter_list:
                        parameters_non_constant += str(key) + ' = ' + str(value) + ' , '
                    else:
                        parameters_constant += str(key) + ' = ' + str(value)

                axes.set_title(parameters_non_constant + '{} best epoch line(s)'.format(best_epoch_lines_to_draw))
                
                #print('Plot - indices:', indices)
                
                axes.plot(self.model_data.get(indices)['loss'], color = 'blue',label = 'Training Loss')
                axes.plot(self.model_data.get(indices)['val_loss'],color='green', label = 'Validation Loss')
                axes.set_ylabel('MSE')
                axes.set_xlabel('Epoch')
                
                # Get indices of smallest values, then sort from smallest to largest to draw the lines
                # in the correct order
                n_smallest_indices = self.get_n_smallest_index_seq(self.model_data.get(indices)['val_loss'],best_epoch_lines_to_draw)
                for index in range(len(n_smallest_indices)):
                    ratio = index / float(len(n_smallest_indices))
                    # Plot vertical lines and create a color gradient
                    axes.axvline(x = n_smallest_indices[index], color = (1.0,0.5*(1.0-ratio),0.5*(1.0-ratio)))
                    value = self.model_data.get(indices)['val_loss'][n_smallest_indices[index]]
                    
                    if index == 0:
                        axes.text(n_smallest_indices[index], self.model_data.get(indices)['val_loss'][n_smallest_indices[index]]*(ratio*5+0.25), str('MSE: {:.4f}'.format(value)), fontsize = 12)
                
                axes.axis([0,len(self.model_data.get(indices)['loss']),0,1.0])
                
                axes.scatter(np.arange(0,len(self.model_data.get(indices)['loss']),1),self.model_data.get(indices)['loss'],color='blue',s = [0.5])
                axes.scatter(np.arange(0,len(self.model_data.get(indices)['loss']),1),self.model_data.get(indices)['val_loss'],color='green', s = [0.5])
                axes.legend()
        
        
        main_title = ''
        if len(parameters_constant) > 0:
            main_title += 'Constant Parameter(s): ' + parameters_constant + ' | '
        main_title += 'Varying Parameter(s): {}'.format(", ".join([parameter for parameter in parameter_name_dict.keys() if parameter in parameter_list]))
            
        plt.suptitle(main_title, size = 16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()
        file_index = parameters_non_constant + '__' + parameters_constant
        file_index = file_index.lower().replace(':','_').replace(' ', '').replace(',','_')
        
        fig.savefig('{}plot-{}.png'.format(self.output_directory_name, file_index))

    def evaluateAll(self, X, y, cv = 10, verbose = True, finalModel = False, param_name_value = None, parameter_values = None):
        
        total_combinations = 1
        for iterator in self.iterators:
            total_combinations *= len(iterator)
        
        if verbose:
            print('\nEvaluating the best network trained for all {} combinations(s)\n'.format(total_combinations))
        
        for model_path in self.modelList:
            
            if finalModel:
                assert(param_name_value is not None), "Provide the param_name_value argument when calling evaluateAll(...) for the final model"
                # Get the last part of the path (filename)
                model_filename = re.search('/([^/]*)$',model_path)
                model_filename = model_filename.group(1)
                # Remove the filename from the path and extract the last part of the path: the configuration.
                model_configuration = 'Final'
                
            else:
                # Get the last part of the path (filename)
                model_filename = re.search('/([^/]*)$',model_path)
                model_filename = model_filename.group(1)
                # Remove the filename from the path and extract the last part of the path: the configuration.
                model_configuration = re.search('/([^/]*)/$', re.sub(r'([^/]*)$', '', model_path))
                model_configuration = model_configuration.group(1)
                
                # Get a list of the values for each parameter 
                parameter_values = model_configuration.split(self.model_parameter_delimiter)
                #parameter_values = model_configuration.split('-')
    
                # Create string, Param1 = Value1,\tParm2 = Value2 etc.
                param_name_value = ",\t".join([self.parameters[i] + ' = ' + parameter_values[i] for i in range(len(self.parameters))])
                
            if verbose:
                print('Evaluating model {} for scenario: {}'.format(model_filename, param_name_value))
                
            K.clear_session() # Bug, kernel dies in Keras 2.2.0 (use 2.1.6 instead)
            
            model = load_model(model_path)
            
            cvscores = TestingFramework.evaluate(model, X, y, cv, verbose)
            
            model_name = model_configuration + '/' + model_filename
            #model_name = model_name.lower().replace(' ','_').replace(',','_')
            self.model_evaluation[model_name] = [np.mean(cvscores), np.std(cvscores), param_name_value, parameter_values, self.parameters]
            
            del model
            
        # Rank all the combinations from lowest mean MSE to highest, with lowest SD
        sorted_combinations = sorted(self.model_evaluation.items(), key=lambda x: x[1])
        #print(sorted_combinations)
        rank = 1
        
        if finalModel:
            self.evaluation_directory_name = self.model_directory_name + self.final_model_directory_name
            os.makedirs(os.path.dirname(self.evaluation_directory_name), exist_ok=True)
            filename = self.evaluation_directory_name + self.evaluation_raking_filename
        else:
            filename = self.evaluation_directory_name + self.evaluation_raking_filename

        for model_score in sorted_combinations:
            rank_string = 'Rank {}\t: {}\t\t| {}\tMSE: {:.3f} +/- {:.3f}\n'.format(rank, model_score[0], model_score[1][2], model_score[1][0], model_score[1][1])
            
            if verbose:
                print(rank_string)
                
            # Write to file
            with open(filename, 'a') as f:
                f.write(rank_string)
                
            rank += 1
        
        return sorted_combinations
            
    @staticmethod
    def getListOfModels(modelDir, modelName = 'best_model.hdf5'):
        modelList = []
        dirData = list(os.walk(modelDir))
        # Loop over all the folders
        for j in dirData[0][1]:
            if modelDir[-1] != '/':
                modelList.append(modelDir + '/' + j + '/' + modelName)
            else:
                modelList.append(modelDir + j + '/' + modelName)
        return modelList
    
    @staticmethod
    def evaluateAllModelsData(baseDir, pathToTestData, pathToTrainingData, 
                              evalDirName = 'Evaluation', modelDirName = 'Models', params = ['Param'], 
                              modelName = 'best_model.hdf5', cv = 10, verbose = True, colRemove = None, 
                              skiprowsTrain = 0, skiprowsTest = 0, param_name_value = None):
                
        if baseDir[-1] != '/':
            baseDir += '/'

        evalDir = baseDir + evalDirName
        modelDir = baseDir + modelDirName
        
        modelList = TestingFramework.getListOfModels(modelDir, modelName = modelName)
        
        total_combinations = len(modelList)
        
        if verbose:
            print('\nEvaluating the best network trained for all {} combinations(s)\n'.format(total_combinations))
        
        modelEvaluation = dict()
        
        X, y = DataHandler.predictionData(pathToTestData, pathToTrainingData, 
                                          skiprowsTrain = skiprowsTrain, skiprowsTest = skiprowsTest,
                                          colRemove=colRemove)

        assert(len(y) > 0), "Error. No truth values were found in the dataset."
        assert(np.shape(X)[0] == np.shape(y)[0]), "Error. The number of columns, {}, in the training data does not equal the number of target truth values, {}.".format(np.shape(X)[0], np.shape(y)[0])
                
        for model_path in modelList:
            
            if param_name_value is None:
                # Get the last part of the path (filename)
                model_filename = re.search('/([^/]*)$',model_path)
                model_filename = model_filename.group(1)
                # Remove the filename from the path and extract the last part of the path: the configuration.
                model_configuration = re.search('/([^/]*)/$', re.sub(r'([^/]*)$', '', model_path))
                model_configuration = model_configuration.group(1)
                
                # Get a list of the values for each parameter 
                #parameter_values = model_configuration.split(self.model_parameter_delimiter)
                parameter_values = model_configuration.split('-')
    
                # Create string, Param1 = Value1,\tParm2 = Value2 etc.
                param_name_value = ",\t".join([params[i%len(params)] + ' = ' + parameter_values[i] for i in range(len(parameter_values))])
            
            if verbose:
                print('Evaluating model {} for scenario: {}'.format(model_filename, param_name_value))
                                          
            K.clear_session() # Bug, kernel dies in Keras 2.2.0

            model = load_model(model_path)
            
            cvscores = TestingFramework.evaluate(model, X, y, cv=cv, verbose=verbose)

            model_name = model_configuration + '/' + model_filename
            #model_name = model_name.lower().replace(' ','_').replace(',','_')
            
            modelEvaluation[model_name] = [np.mean(cvscores), np.std(cvscores), param_name_value]
            
            del model
            
                        
        # Rank all the combinations from lowest mean MSE to highest
        sorted_combinations = sorted(modelEvaluation.items(), key=lambda x: x[1])
        #print(sorted_combinations)
        
        rank = 1
        filename = evalDir + '/' + 'ranked_evaluation_static.txt'

        for model_score in sorted_combinations:
            rank_string = 'Rank {}\t: {}\t\t| {}\tMSE: {:.3f} +/- {:.3f}\n'.format(rank, model_score[0], model_score[1][2], model_score[1][0], model_score[1][1])
            
            if verbose:
                print(rank_string)
                
            # Write to file
            with open(filename, 'a') as f:
                f.write(rank_string)
                
            rank += 1
    
    @staticmethod
    def evaluate(model, X, y, cv = 10, verbose = True):
                
        kfold = KFold(n_splits=cv, shuffle=True)
        cvscores = []
        
        for _, test in kfold.split(X, y):
            scores = model.evaluate(X[test], y[test], verbose=0)
            #print("{}: {:.2f}".format(model.metrics_names[1], scores[1]))
            cvscores.append(scores[1])
                
        if verbose:
            print('MSE: {:.2f} +/- {:.2f}\n'.format(np.mean(cvscores), np.std(cvscores)))
            
        return cvscores
        
    @staticmethod
    def predict(pathToModel, pathToTestData, pathToTrainingData, verbose = True, colRemove = None, skiprowsTest = 0, skiprowsTrain = 0):
        
        K.clear_session()
        model = load_model(pathToModel)

        X, y = DataHandler.predictionData(pathToTestData, pathToTrainingData, 
                                          skiprowsTest = skiprowsTest, skiprowsTrain = skiprowsTrain,
                                          colRemove=colRemove)
        _, y_scaler = DataHandler.getXYTransformer(pathToTrainingData, colRemove = colRemove, skiprows = skiprowsTrain)
        
        y_scale = y_scaler.scale_
        y_mean = y_scaler.mean_
        
        y_pred = model.predict(X)
        
        y_pred = y_pred*y_scale + y_mean
        y = y*y_scale + y_mean
        
        mse = np.sum((np.array(y)-np.array(y_pred))**2/(len(y)))
        
        if verbose:
            print("Predictions\t\tActual")
            for i in range(len(y_pred)):
                print("{:.2f}".format(y_pred[i][0]), end='')
                try:
                    print("\t\t\t{:.2f}".format(y[i][0]))
                except:
                    print("\t\t\tMissing Value")
                    pass
        print('MSE: {:.4f}\t\tRMSE: {:.4f}'.format(mse, np.sqrt(mse)))
        
        del model
        
        if len(y) == len(y_pred):
            y_func = lambda x: x
            x_range = np.arange(1,max(np.max(y_pred),np.max(y))+0.1,0.1)
            plt.figure()
            plt.scatter(y_pred, y)
            plt.plot(x_range, y_func(x_range), color = 'green')
            plt.xlabel('MOS - Prediction')
            plt.ylabel('MOS - Actual')
            plt.xticks(np.arange(1,4+0.2,0.2))
            plt.yticks(np.arange(1,4+0.2,0.2))
            plt.grid(linestyle='--')

        
    def get_n_smallest_index_seq(self, array, n):
        """ 
        Returns the n smallest values in an array with the condition that for each value in the list
        there are (index) values infront/behind the current value that is lower.
        
        That is, the number with index 0 in the returned array has 0 values smaller in front of it,
        the number with index 1 has one value that is smaller infront etc.
        
        Given an array, arr = [10,3,5,4]
        
        get_n_smallest_index_seq(arr, 2) -> [1, 0]
        
        """
        
        assert (n <= len(array)), "Number of values cannot be larger than elements in array"
        
        n_smallest_seq = []        
        smallest = np.inf
        for i in range(len(array)):
            if array[i] < smallest:
                n_smallest_seq.append(i)
                smallest = array[i]
        
        # Return only the n last indices
        # Ensure the smallest index is first in the returned array
        return n_smallest_seq[-1:-1*(n+1):-1]
    
    
    def gridsearch(self, dataHandler, **parameters_values):
        
        features = dataHandler.features
    
        print('Running with {} features'.format(features))
        
        X_train, y_train, X_test, y_test, X_val_1, y_val_1, X_val_2, y_val_2 = dataHandler.getSplitData(validationData = True, normalizeData = True)
        #print(parameters_values)
        
        self.train(X_train, y_train, X_val = X_val_1, y_val = y_val_1, 
                   patience = 100, finalModel = False,
                   **parameters_values
                  )
        
        self.plotAll(scale = 4.0)
        sorted_list = self.evaluateAll(X_val_2, y_val_2, cv = 10, verbose = False)
        
        # Get the best hyperparameters
        best_model = sorted_list[0]
        #print(best_model)
        
        # Train the final model based on the hyper parameter search
        X_train_final = np.concatenate((X_train, X_val_2))
        y_train_final = np.concatenate((y_train, y_val_2))
        
        self.reset()
        self.trainFinalModel(X_train_final, y_train_final, X_test, y_test, X_val_1, y_val_1, best_model)
        
    def trainFinalModel(self, X_train, y_train, X_test, y_test, X_val, y_val, best_model):
        
        # best_model is a list containing the following:
        # (Index : item)
        # 0  : (float) Mean cvscores         |       1 : (float) std cvscores 
        # 2  : (str) param name + value      |       3 : (list of ints) parameter values
        # 4  : (str) name of parameters
        
        training_parameters = dict()
        parameter_names = best_model[1][4]
        parameter_values = best_model[1][3]
        for i in range(len(parameter_names)):
            value = parameter_values[i]
            # Needs to converted to int or float based on underlying type in the str
            training_parameters['param_{}'.format(i)] = {parameter_names[i] : [float(value) if '.' in value else int(value)]}
        
        #print('Training_Parameters: ', training_parameters)
        model_history = self.train(X_train, y_train, X_val = X_val, y_val = y_val, 
           patience = 1000, finalModel = True,
           **training_parameters)
    
        #model_history = self.model_data.get(parameter_values)
    
        ###
        # Plot the loss curve for the final model
        plt.figure(figsize=(15,15))
        best_epoch_lines_to_draw = 10
        plt.title(best_model[1][2] + ' | {} best epoch line(s)'.format(best_epoch_lines_to_draw))
        
        #print('Plot - indices:', indices)
        
        plt.plot(model_history['loss'], color = 'blue',label = 'Training Loss')
        plt.plot(model_history['val_loss'],color='green', label = 'Testing Loss')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        
        # Get indices of smallest values, then sort from smallest to largest to draw the lines
        # in the correct order
        n_smallest_indices = self.get_n_smallest_index_seq(model_history['val_loss'], best_epoch_lines_to_draw)
        for index in range(len(n_smallest_indices)):
            ratio = index / float(len(n_smallest_indices))
            # Plot vertical lines and create a color gradient
            plt.axvline(x = n_smallest_indices[index], color = (1.0,0.5*(1.0-ratio),0.5*(1.0-ratio)))
            value = model_history['val_loss'][n_smallest_indices[index]]
            
            if index == 0:
                plt.text(n_smallest_indices[index], model_history['val_loss'][n_smallest_indices[index]]*(ratio*5+0.25), str('MSE: {:.4f}'.format(value)), fontsize = 12)
        
        #plt.axis([0,n_smallest_indices[0]*1.1,0,0.2])
        plt.axis([0,len(model_history['val_loss']),0,0.5])

        plt.scatter(np.arange(0,len(model_history['loss']),1),model_history['loss'],color='blue',s = [0.5])
        plt.scatter(np.arange(0,len(model_history['loss']),1),model_history['val_loss'],color='green', s = [0.5])
        plt.legend()
        
        plt.savefig('{}-final-plot-1.png'.format(self.output_directory_name))
        ###

        self.evaluateAll(X_test, y_test, cv = 10, verbose = False, 
                         finalModel = True, param_name_value = best_model[1][2],
                         parameter_values = parameter_values)



 
