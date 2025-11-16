import pickle, json
from jules import _input_transform, train_surrogate, infer_surrogate, optimise_response

TRAIN_DATA_ = json.dumps({"101":{"parameters": {"param1": 1, "param2": 1, "param3": 1}, "responses" : {"resp1": 100, "resp2": 0}},
"102": {"parameters": {"param1": 1, "param2": 1, "param3": 2},"responses" : {"resp1": 110, "resp2": 3}},
"103": {"parameters": {"param1": 1, "param2": 2, "param3": 1},"responses" : {"resp1": 125, "resp2": 6}},
"104": {"parameters": {"param1": 1, "param2": 2, "param3": 2},"responses" : {"resp1": 135, "resp2": 9}},
"105": {"parameters": {"param1": 2, "param2": 1, "param3": 1},"responses" : {"resp1": 90, "resp2": 4}},
"106": {"parameters": {"param1": 2, "param2": 1, "param3": 2},"responses" : {"resp1": 100, "resp2": 2}},
"107": {"parameters": {"param1": 2, "param2": 2, "param3": 1},"responses" : {"resp1": 115, "resp2": 3}},
"108": {"parameters": {"param1": 2, "param2": 2, "param3": 2},"responses" : {"resp1": 125, "resp2": 1}}})

TEST_DATA_ = json.dumps({"0":{"parameters": {"param1": 1, "param2": 1, "param3": 1}, "responses" : {} }})

if __name__ == "__main__":
    parameters, responses = _input_transform(TRAIN_DATA_)
    model_str = train_surrogate(parameters, responses)
    
    model = pickle.loads(model_str)
    parameters, responses = _input_transform(TEST_DATA_)
    infer_surrogate(parameters, model)

    optimise_response("resp1", model)


