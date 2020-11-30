# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd

def generation(num_samples, num_frames):
    seed = 123
    #num_samples = 5
    exposure_time = 33.0e-3
    interval = 33.0e-3
    #num_frames = 5
    Nm = [100, 100, 100]
    Dm = [0.222e-12, 0.032e-12, 0.008e-12]
    transmat = [
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.2],
        [0.0, 1.0, 0.0]]

    import numpy
    rng = numpy.random.RandomState(seed)
    import scopyon
    config = scopyon.DefaultConfiguration()
    config.default.effects.photo_bleaching.switch = False
    config.default.detector.exposure_time = exposure_time
    pixel_length = config.default.detector.pixel_length / config.default.magnification
    L_2 = config.default.detector.image_size[0] * pixel_length * 0.5
    
    timepoints = numpy.linspace(0, interval * num_frames, num_frames + 1)
    ndim = 2

    import pathlib
    # TODO: set appropriate path
    artifacts = pathlib.Path("01_raw/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    # 
    config.save(artifacts / 'config.yaml')
    
    #
    for i in range(num_samples):
        samples = scopyon.sample(timepoints, N=Nm, lower=-L_2, upper=+L_2, ndim=ndim, D=Dm, transmat=transmat, rng=rng)
        inputs = [(t, numpy.hstack((points[:, : ndim], points[:, [ndim + 1]], numpy.ones((points.shape[0], 1), dtype=numpy.float64)))) for t, points in zip(timepoints, samples)]
        ret = list(scopyon.generate_images(inputs, num_frames=num_frames, config=config, rng=rng, full_output=True))

        inputs_ = []
        for t, data in inputs:
            inputs_.extend(([t] + list(row) for row in data))
        inputs_ = numpy.array(inputs_)
        numpy.save(artifacts / f"inputs{i:03d}.npy", inputs_)

        numpy.save(artifacts / f"images{i:03d}.npy", numpy.array([img.as_array() for img, infodict in ret]))

        true_data = []
        for t, (_, infodict) in zip(timepoints, ret):
            true_data.extend([t, key] + list(value) for key, value in infodict['true_data'].items())
        true_data = numpy.array(true_data)
        numpy.save(artifacts / f"true_data{i:03d}.npy", true_data)

def split_data(data: pd.DataFrame, example_test_data_ratio: float) -> Dict[str, Any]:
    """Node for splitting the classical Iris data set into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.
    """
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    classes = sorted(data["target"].unique())
    # One-hot encoding for the target variable
    data = pd.get_dummies(data, columns=["target"], prefix="", prefix_sep="")

    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * example_test_data_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.loc[:, "sepal_length":"petal_width"]
    train_data_y = training_data[classes]
    test_data_x = test_data.loc[:, "sepal_length":"petal_width"]
    test_data_y = test_data[classes]

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y,
    )
