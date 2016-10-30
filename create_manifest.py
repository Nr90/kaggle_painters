import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder


dataset = 'data/train_info.csv'
output = 'data/manifest.csv'

in_dataframe = pandas.read_csv(dataset)
filenames = in_dataframe['filename'].values
painters = in_dataframe['artist'].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(painters)
painter = encoder.transform(painters)

out_dataframe = pandas.DataFrame({'filenames': filenames, 'painter': painter})
out_dataframe.to_csv(output, index=False)
print 'Number of different painters: %s' % (np.amax(painter) + 1)
