"""
Author a007010 : Yohann GOARDOU
I am probably not working at Renault Group anymore.
This file intend to propose services for Machine Learning in CAE context, in particular for application such as Stylise
I called it JULES : Just Use Linear Elastic Surrogate
For obvious reasons

The script is to be called as follow :
jules.py --action [train|infer|optimise] [--experiments] [--model] [--target]
"""
import argparse, json, pickle

import pandas as pd

from scikit-learn import preprocessing, pipeline, multioutput, linear_model, ensemble
from scipy import optimize

class MetaData:
    def __init__(self, parameters: pd.DataFrame, responses: pd.DataFrame):
        self.parameters = parameters.describe()  # DataFrame, includes min and max values (used for optimisation bounds)
        self.responses = responses.columns  # Series, for names only


def _input_transform(input_from_server: str) -> pd.DataFrame:
    """
    This function translates a DoE from the backend to a pandas DataFrame to be used by the services
    Expected input format:
    {
        exp_id_1:
            {
                parameters: {param1: value1_1, param2: value2_1, param3: value3_1},
                responses : {resp1: value1_1, resp2: value2_1}
            },
        exp_id_2: 
            {
                parameters: {param1: value1_2, param2: value2_2, param3: value3_2},
                responses : {resp1: value1_2, resp2: value2_2}
            } 
    }
    """
    data = dict(json.loads(input_from_server))
    parameters = pd.concat([pd.DataFrame(values['parameters'], index=[key]) for key, values in data.items()])
    responses = pd.concat([pd.DataFrame(values['responses'], index=[key]) for key, values in data.items()])
    return parameters, responses


def _standard_pipeline() -> pipeline.Pipeline:
    scaler = preprocessing.StandardScaler()
    expert_mixture = ensemble.StackingRegressor(estimators=[
        ("elasticnetlin", linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], )),
        ("elasticnetpoly", pipeline.Pipeline([
            ("poly", preprocessing.PolynomialFeatures()), 
            ("e_net_p", linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1] ))]))
    ], final_estimator=linear_model.LinearRegression())
    pipe = pipeline.Pipeline([
        ("scaler", scaler),
        ("expert_mixture", multioutput.MultiOutputRegressor(expert_mixture))
    ])
    return pipe


def train_surrogate(parameters_df: pd.DataFrame, responses_df: pd.DataFrame) -> None:
    fitted_pipeline = _standard_pipeline().fit(parameters_df, responses_df)  # Train the model using the standard pipeline
    fitted_pipeline.metadata = MetaData(parameters_df, responses_df)  # Add the parameters/responses data to access them in infer and optimisation mode
    pipe_str = pickle.dumps(fitted_pipeline, protocol=5)
    print(pipe_str)  # print the model in a string format, that can be reloaded and used in the future
    return pipe_str


def infer_surrogate(parameters_df: pd.DataFrame, model: pipeline.Pipeline) -> None:
    prediction = model.predict(parameters_df)  # Predict the responses using the trained pipeline
    infer_df = pd.DataFrame(prediction, columns=model.metadata.responses)
    # infer_str = {response: prediction[i] for i, response in enumerate(model.metadata.responses)}
    print(infer_df)  # print the values as a dict to be processed by the backend
    return infer_df
    

def optimise_response(resp: str, model: pipeline.Pipeline) -> None:
    bounds = [(model.metadata.parameters.loc['min', param], model.metadata.parameters.loc['max', param]) for param in model.metadata.parameters]
    
    def objective_function(x):
        x_df = pd.DataFrame(x.reshape(1, -1), columns=model.metadata.parameters.columns)
        predictions = pd.DataFrame(model.predict(x_df).reshape(1, -1), columns=model.metadata.responses)
        resp_pred = predictions.loc[0, resp]
        return resp_pred
    
    res_shgo = optimize.shgo(objective_function, bounds)
    if res_shgo.success:
        optim_ret = {param: float(res_shgo.x[i]) for i, param in enumerate(model.metadata.parameters.columns)}
    else:
        optim_ret = None

    print(optim_ret)
    return optim_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JULES: Just Use Linear Elastic Surrogate",
        epilog="For styLISE")
    
    # Logics of arguments is here to decide what to do with JULES
    parser.add_argument("--action", help="select the action to perform", choices=['train', 'infer', 'optimise'], required=True)

    # Additional arguments required to perform the desired task
    parser.add_argument("-x", "--experiments", help="json containing experimental data to train on or infer") # required for train and infer options
    parser.add_argument("-m", "--model", help="stringified version of the model returned by train")  # required for infer and optimise options
    parser.add_argument("-t", "--target", help="name of the response to minimize")  # required for optimise

    args = parser.parse_args()

    if args.action == 'train':
        parameters, responses = _input_transform(args.experiments)
        train_surrogate(parameters, responses)
    elif args.action == 'infer':
        parameters, _ = _input_transform(args.experiments)
        model = pickle.loads(args.model)
        infer_surrogate(parameters, model)
    elif args.action == 'optimise':
        model = pickle.loads(args.model)
        optimise_response(args.target, model)
