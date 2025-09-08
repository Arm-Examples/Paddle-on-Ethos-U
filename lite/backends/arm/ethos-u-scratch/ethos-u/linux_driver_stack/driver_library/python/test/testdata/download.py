#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
from pathlib import Path
from typing import List
from urllib.request import urlopen
"""
Downloads resources for tests from Arm public model zoo.
Run this script before executing tests.
"""


PMZ_URL = 'https://github.com/ARM-software/ML-zoo/raw/9f506fe52b39df545f0e6c5ff9223f671bc5ae00/models'
test_resources = [
    {'model': '{}/visual_wake_words/micronet_vww2/tflite_int8/vww2_50_50_INT8.tflite'.format(PMZ_URL),
     'ifm': '{}/visual_wake_words/micronet_vww2/tflite_int8/testing_input/input/0.npy'.format(PMZ_URL),
     'ofm': '{}/visual_wake_words/micronet_vww2/tflite_int8/testing_output/Identity/0.npy'.format(PMZ_URL)}
]


def download(path: str, url: str):
    with urlopen(url) as response, open(path, 'wb') as file:
        print("Downloading {} ...".format(url))
        file.write(response.read())
        file.seek(0)
        print("Finished downloading {}.".format(url))


def download_test_resources(test_res_entries: List[dict], where_to: str):
    os.makedirs(where_to, exist_ok=True)

    for resources in test_res_entries:
        download(os.path.join(where_to, 'model.tflite'), resources['model'])
        download(os.path.join(where_to, 'model_ifm.npy'), resources['ifm'])
        download(os.path.join(where_to, 'model_ofm.npy'), resources['ofm'])


def main():
    current_dir = str(Path(__file__).parent.absolute())
    download_test_resources(test_resources, os.path.join(current_dir, 'shared'))


if __name__ == '__main__':
    main()
