import pandas as pd
import os
import opendatasets as od

dataset = "https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256"

od.download(dataset)

