import os
import numpy as np
import csv
from collections import OrderedDict
from collections import Iterable
import numpy as np
from keras import backend as K
from keras.callbacks import Callback, CSVLogger

def coef_det_k(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def coef_det_k_2(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - SS_res / (SS_tot + 1e-6)

class TestCallback(Callback): 
    def __init__(self, test_data): 
        self.test_data = test_data
        self.loss = -1e8
        self.acc = -1e8
    
    def on_epoch_end(self, epoch, logs={}): 
        x, y = self.test_data 
        self.loss, self.acc = self.model.evaluate(x, y, verbose=0) 
        
        

class MyCSVLogger(CSVLogger):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
    ```python
    csv_logger = my_CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        hpars: dictionary with additional values to store, e.g. current hyperparameters
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, test=None, separator=',', append=False):

        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.test = test
        self.file_flags = ''
        self._open_args = {'newline': '\n'}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.test is not None:
            self.test_dict = {'test_loss': self.test.loss,'test_val_det_k': self.test.acc} 
            logs = {**logs,  **self.test_dict}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
