import pandas as pd

data = {
    'StudyHours': {},
    'Attendance': {},
    'Actual': {},
}

df = pd.DataFrame(data)
x = df[['StudyHours', 'Attendance']]
y = df['Actual']