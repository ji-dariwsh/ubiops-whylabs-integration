"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import pickle
import pandas as pd
from whylogs import get_or_create_session


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        :param str base_directory: absolute path to the directory where the deployment.py file is located
        :param dict context: a dictionary containing details of the deployment that might be useful in your code.
            It contains the following keys:
                - deployment (str): name of the deployment
                - version (str): name of the version
                - input_type (str): deployment input type, either 'structured' or 'plain'
                - output_type (str): deployment output type, either 'structured' or 'plain'
                - language (str): programming language the deployment is running
                - environment_variables (str): the custom environment variables configured for the deployment.
                    You can also access those as normal environment variables via os.environ
        """

        print("Initialising the model")
        model_file_name = "model.pkl"
        model_file = os.path.join(base_directory, model_file_name)
        self.wl_session = get_or_create_session()
        with open(model_file, 'rb') as file:
            self.model = pickle.load(file)

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.

        :param dict/str data: request input data. In case of deployments with structured data, a Python dictionary
            with as keys the input fields as defined upon deployment creation via the platform. In case of a deployment
            with plain input, it is a string.
        :return dict/str: request output. In case of deployments with structured output data, a Python dictionary
            with as keys the output fields as defined upon deployment creation via the platform. In case of a deployment
            with plain output, it is a string. In this example, a dictionary with the key: output.
        """
        print('Loading data')
        X = pd.read_csv(data['data'])

        # Log input data to whylabs
        self.wl_session.log_dataframe(X, 'model.input.data')

        print("Prediction being made")
        prediction = self.model.predict(X)
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['target'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv',
        }
