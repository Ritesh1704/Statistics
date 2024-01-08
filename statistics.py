import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
# Mean
x = [2,4,6,7,20,10,22]
np.mean(x)
# Median
x = [2,4,6,7,20,10,22]
np.median(x)
# Mode
from statistics import mode
mode([3,3,1,12,1,2,2,4,4])
# Range
data = [2,4,5,7,20,10,22]
np.ptp(data)
# Variance
data = [160, 162, 170, 178, 165, 175, 168]
np.var(data)
# Std Deviation
data = [160, 162, 170, 178, 165, 175, 168]
np.std(data)
# Applying these statistics to a dataset
df = pd.read_csv('student.csv')
df
df.shape
df['maths_score'].describe()
import matplotlib.pyplot as plt
plt.figure(figsize = (25,7))
plt.boxplot(df['maths_score'], vert=False)
plt.xticks(np.arange(10, 110, 2))
plt.grid()
plt.show()
desc_marks = df['maths_score'].describe()
desc_marks
# Calculate the Interquartile Range
Q1 = desc_marks['25%']
Q3 = desc_marks['75%']
IQR = Q3 - Q1
IQR
# Calculate the UpperRange and the Lower Range
LwrRange = Q1 - (1.5 * IQR)
UprRange = Q3 + (1.5 * IQR)
LwrRange, UprRange
# Who are my outliers
df.loc[(df['maths_score'] <= LwrRange) | (df['maths_score'] >= UprRange)]
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(df, x="maths_score", kde=True)
plt.grid()
plt.show()
![image.png](attachment:image.png)
# 1st Std Deviation = 66.5 + 16.5 = 83 / 66.5 - 16.5 = 50
# 2nd Std Deviation = 83 + 16.5 = 99.5 / 50 - 16.5 = 33.5
