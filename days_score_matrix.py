import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv('pilot0.txt', sep="|", decimal=",", na_values="", skiprows=0, encoding='latin1', index_col=False, header=None)
data.columns = ['roomId','date_time','comment','rating','ratingId'] # naming the columns
data['date'] =data.date_time.str.split(' ').str.get(1)  # remove time, retaining only date
data = data[['roomId','date','rating']] # discarding the irrelevant columns

# the matrix generation
num_score = len(set(data.rating))
num_days = len(set(data.date))
count = data.groupby(['date','rating']).size()
c = count.unstack(level=1).fillna(0.0)
c.to_csv('frequency_counter.csv')
cmat = c.as_matrix() # converting to matrix
cnorm = cmat/max(cmat.max(axis=1)) # normalizing for plotting
date_labels  =[x.encode('utf-8') for x in count.index.levels[0].values]
plt.imshow(cnorm)
plt.xlabel(date_labels)
plt.show()