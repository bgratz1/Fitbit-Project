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

#plt.show()
#plt.pause(5)            # keep it open for 5 seconds
#plt.close()             # close the plot


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
#plt.show()
#plt.pause(5)            # keep it open for 5 seconds
#plt.close()             # close the plot


#doing time block analysis

#steps per block
query2=f'SELECT Id,ActivityHour,StepTotal FROM hourly_steps'

cursor=connection.cursor()
cursor.execute(query2)
rows = cursor.fetchall()

step_df=pd.DataFrame(rows,columns=['Id','ActivityHour','StepTotal'])

step_df['ActivityHour']=pd.to_datetime(step_df['ActivityHour'])

#want just hour
step_df['hour']=step_df['ActivityHour'].dt.hour

#defining a function to use on steps, cals burnt and mins of sleep
def blocks(hour):
    if 0<=hour<4:
        return '0-4'
    elif 4<=hour<8:
        return '4-8'
    elif 8<=hour<12:
        return '8-12'
    elif 12<=hour<16:
        return '12-16'
    elif 16<=hour<20:
        return '16-20'
    else:
        return '20-24'

step_df['block']=step_df['hour'].apply(blocks)

avg_steps=step_df.groupby('block')['StepTotal'].mean()

order=['0-4','4-8','8-12','12-16','16-20','20-24']
avg_steps=avg_steps.reindex(order)

plt.figure()
plt.bar(avg_steps.index, avg_steps.values)
plt.xlabel("blocks")
plt.ylabel("average steps")
plt.title("average steps taken in each block")
#plt.show()


#doing similar for cals burnt
query3=f'SELECT Id,ActivityHour,Calories FROM hourly_calories'

cursor=connection.cursor()
cursor.execute(query3)
rows = cursor.fetchall()

cal_df=pd.DataFrame(rows,columns=['Id','ActivityHour','Calories'])

cal_df['ActivityHour']=pd.to_datetime(cal_df['ActivityHour'])

#want just hour
cal_df['hour']=cal_df['ActivityHour'].dt.hour

cal_df['block']=cal_df['hour'].apply(blocks)

avg_cals=cal_df.groupby('block')['Calories'].mean()

avg_cals=avg_cals.reindex(order)

plt.figure()
plt.bar(avg_cals.index, avg_cals.values)
plt.xlabel("blocks")
plt.ylabel("average cals burned")
plt.title("average cals burned in each block")
#plt.show()


#doing similar for mins of sleep

#want just hour
sleep_df['hour']=sleep_df['date'].dt.hour
sleep_df['block']=sleep_df['hour'].apply(blocks)

sleep_block=sleep_df.groupby(['Id','block']).size().reset_index(name='sleepMins')

avg_sleep=sleep_block.groupby('block')['sleepMins'].mean()

avg_sleep=avg_sleep.reindex(order)

plt.figure()
plt.bar(avg_sleep.index, avg_sleep.values)
plt.xlabel("blocks")
plt.ylabel("average sleep")
plt.title("average sleep in each block")
#plt.show()



#defining a fxn for ID as input
def plot_heart(Id):
    query1=f'SELECT Id,Time,Value FROM heart_rate WHERE Id={Id}'
    cursor=connection.cursor()  
    cursor.execute(query1)
    rows_heart = cursor.fetchall()

    query2=f'SELECT Id,ActivityHour,TotalIntensity FROM hourly_intensity WHERE Id={Id}'
    cursor=connection.cursor()  
    cursor.execute(query2)
    rows_intens = cursor.fetchall()

    heart_df=pd.DataFrame(rows_heart,columns=['Id','Time','Value'])
    intens_df=pd.DataFrame(rows_intens,columns=['Id','ActivityHour','TotalIntensity'])

    heart_df['Time']=pd.to_datetime(heart_df['Time'])
    intens_df['ActivityHour']=pd.to_datetime(intens_df['ActivityHour'])

    fig,ax=plt.subplots(2,1)
    ax[0].plot(heart_df['Time'],heart_df['Value'])
    ax[0].set_title(f'heart rate for Id {Id}')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('heart rate')

    ax[1].plot(intens_df['ActivityHour'],intens_df['TotalIntensity'])
    ax[1].set_title(f'total intensity for Id {Id}')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('tot intensity')

    return fig
print(pd.read_sql_query("SELECT DISTINCT Id FROM heart_rate", connection).sample(1)['Id'].iloc[0])
fig=plot_heart(6391747486)
#plt.show()
