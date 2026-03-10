import pandas as pd
import sqlite3
from datetime import datetime
from scipy import stats

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


#figuring out the daily total sleep
grouped=sleep_df.groupby(['Id','sleepDay'])
counts=grouped.size()
sleep_daily=counts.reset_index()
sleep_daily=sleep_daily.rename(columns={0:'sleepMins'})


df['activeMins']=df['VeryActiveMinutes']+df['FairlyActiveMinutes']+df['LightlyActiveMinutes']

#creating final df with sleeptotals
final=pd.merge(df,sleep_daily,left_on=['Id','ActivityDate'],right_on=['Id','sleepDay'],how='inner')


result=stats.linregress(final['activeMins'],final["sleepMins"])
print("Slope:", result.slope)
print("Intercept:", result.intercept)
print("R^2:", result.rvalue**2)
print("p-value:", result.pvalue)
