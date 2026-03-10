import pandas as pd
import sqlite3
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("daily_acivity.csv")


#making three classes of people
counts=df['Id'].value_counts()

class_df=counts.reset_index()
class_df.columns=['Id','freq']

class_df['Class']='light'
class_df.loc[class_df['freq']>10,'Class']='moderate'
class_df.loc[class_df['freq']>15,'Class']='heavy'
class_df=class_df[['Id','Class']]


connection=sqlite3.connect('fitbit_database.db')
#this query gives the minutes slept per user ID
query=f'SELECT Id,date FROM minute_sleep'

cursor=connection.cursor()
cursor.execute(query)
rows = cursor.fetchall()
connection.close()
sleep_df=pd.DataFrame(rows,columns=['Id','date'])

## used chatGPT to figure out how to isolate the day from the time the db gave in text
sleep_df['date']=pd.to_datetime(sleep_df['date'])
df['ActivityDate']=pd.to_datetime(df['ActivityDate'])
sleep_df['sleepDay']=sleep_df['date'].dt.normalize()
df['ActivityDate']=df['ActivityDate'].dt.normalize()
##

grouped=sleep_df.groupby(['Id','sleepDay'])
counts=grouped.size()
sleep_daily=counts.reset_index()
sleep_daily=sleep_daily.rename(columns={0:'sleepMins'})


df['activeMins']=df['VeryActiveMinutes']+df['FairlyActiveMinutes']+df['LightlyActiveMinutes']

df['Id'] = df['Id'].astype('int64')
sleep_daily['Id'] = sleep_daily['Id'].astype('int64')
final=pd.merge(df,sleep_daily,left_on=['Id','ActivityDate'],right_on=['Id','sleepDay'],how='inner')
print(final)

result1=stats.linregress(final['activeMins'],final["sleepMins"])
print("Slope:", result1.slope)
print("Intercept:", result1.intercept)
print("R^2:", result1.rvalue**2)
print("p-value:", result1.pvalue)

plt.scatter(final['activeMins'], final['sleepMins'])

# regression line
x = final['activeMins']
y = result1.slope * x + result1.intercept
plt.plot(x, y)

plt.xlabel("Active Minutes")
plt.ylabel("Sleep Minutes")
plt.title("Relationship Between Activity and Sleep")

plt.show()
plt.pause(5)            # keep it open for 5 seconds
plt.close()             # close the plot


#doing same analysis for sedentary mins
print(df['SedentaryMinutes'].shape)
print(final['sleepMins'].shape)
result2=stats.linregress(final['SedentaryMinutes'],final["sleepMins"])
print("Slope:", result2.slope)
print("Intercept:", result2.intercept)
print("R^2:", result2.rvalue**2)
print("p-value:", result2.pvalue)

plt.scatter(final['SedentaryMinutes'], final['sleepMins'])

# regression line
x = final['SedentaryMinutes']
y = result2.slope * x + result2.intercept
plt.plot(x, y)

plt.xlabel("sedentary Minutes")
plt.ylabel("Sleep Minutes")
plt.title("Relationship Between sedentary and Sleep")
plt.show()
plt.pause(5)            # keep it open for 5 seconds
plt.close()             # close the plot


#doing time block analysis

#steps per block
