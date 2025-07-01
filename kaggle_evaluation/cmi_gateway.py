"""Gateway notebook for https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data"""

import os
import random
import time

import polars as pl

import kaggle_evaluation.core.templates

from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType


class CMIGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths=data_paths, target_column_name='gesture')
        self.set_response_timeout_seconds(30 * 60)
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_gestures = self.target_gestures + self.non_target_gestures

    def unpack_data_paths(self) -> None:
        if not self.data_paths:
            self.test_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
            self.demographics_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
        else:
            self.test_path = self.data_paths[0]
            self.demographics_path = self.data_paths[1]

    def generate_data_batches(self):
        test = pl.read_csv(self.test_path)
        demos = pl.read_csv(self.demographics_path)

        # Different order for every run.
        # Using time as the seed is the default behavior but it's so important that we'll be explicit.
        random.seed(time.time())

        sequence_ids = list(test['sequence_id'].unique())
        for seq_id in random.sample(sequence_ids, len(sequence_ids)):
            sequence = test.filter(pl.col('sequence_id') == seq_id)
            sequence_demos = demos.filter(pl.col('subject') == sequence['subject'][0])
            yield (sequence, sequence_demos), pl.DataFrame(data={'sequence_id':[seq_id]})

    def validate_prediction_batch(self, prediction: str, row_ids: pl.Series) -> None:
        if not isinstance(prediction, str):
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Invalid prediction type, expected str but got {type(prediction)}'
            )
        if prediction not in self.all_gestures:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'All gestures must match those found in the train data. Got {prediction}'
            )
        super().validate_prediction_batch(prediction, row_ids)


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = CMIGateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print('Skipping run for now')
