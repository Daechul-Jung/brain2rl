import os
import sys

import kaggle_evaluation.core.templates

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import cmi_gateway


class CMIInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return cmi_gateway.CMIGateway(data_paths)
