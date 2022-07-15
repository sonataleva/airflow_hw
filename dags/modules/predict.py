import os
import pandas as pd
import dill
import sklearn
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')

def get_latest_model():
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]

    with open(f'{path}/data/models/{latest_model}', 'rb') as file:
        model = dill.load(file)

    return model

def predict():
    best_pipe = get_latest_model()
    list_jsons = [
        '7310993818',
        '7313922964',
        '7315173150',
        '7316152972',
        '7316509996'
    ]

    predictions = {}
    for object in list_jsons:
        predictions[object] = best_pipe.predict(
            pd.DataFrame(pd.read_json(f'{path}/data/test/{object}.json', typ='series')).T)

    df = pd.DataFrame.from_dict(predictions, orient='index', columns=['pred'])

    return df.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')



if __name__ == '__main__':
    predict()
