import pandas as pd

df1 = pd.read_csv('datasets/amsterdam_weekdays.csv', sep=',')
df1['weekend'] = False
df2 = pd.read_csv('datasets/amsterdam_weekends.csv', sep=',')
df2['weekend'] = True
amsterdam = pd.concat([df1, df2], ignore_index=True, sort=False)
amsterdam['city'] = 'Amsterdam'

df3 = pd.read_csv('datasets/athens_weekdays.csv', sep=',')
df3['weekend'] = False
df4 = pd.read_csv('datasets/athens_weekends.csv', sep=',')
df4['weekend'] = True
athens = pd.concat([df3, df4], ignore_index=True, sort=False)
athens['city'] = 'Athens'

df5 = pd.read_csv('datasets/barcelona_weekdays.csv', sep=',')
df5['weekend'] = False
df6 = pd.read_csv('datasets/barcelona_weekends.csv', sep=',')
df6['weekend'] = True
barcelona = pd.concat([df5, df6], ignore_index=True, sort=False)
barcelona['city'] = 'Barcelona'

df7 = pd.read_csv('datasets/berlin_weekdays.csv', sep=',')
df7['weekend'] = False
df8 = pd.read_csv('datasets/berlin_weekends.csv', sep=',')
df8['weekend'] = True
berlin = pd.concat([df7, df8], ignore_index=True, sort=False)
berlin['city'] = 'Berlin'

df9 = pd.read_csv('datasets/budapest_weekdays.csv', sep=',')
df9['weekend'] = False
df10 = pd.read_csv('datasets/budapest_weekends.csv', sep=',')
df10['weekend'] = True
budapest = pd.concat([df9, df10], ignore_index=True, sort=False)
budapest['city'] = 'Budapest'

df11 = pd.read_csv('datasets/lisbon_weekdays.csv', sep=',')
df11['weekend'] = False
df12 = pd.read_csv('datasets/lisbon_weekends.csv', sep=',')
df12['weekend'] = True
lisbon = pd.concat([df11, df12], ignore_index=True, sort=False)
lisbon['city'] = 'Lisbon'

df13 = pd.read_csv('datasets/london_weekdays.csv', sep=',')
df13['weekend'] = False
df14 = pd.read_csv('datasets/london_weekends.csv', sep=',')
df14['weekend'] = True
london = pd.concat([df13, df14], ignore_index=True, sort=False)
london['city'] = 'London'

df15 = pd.read_csv('datasets/paris_weekdays.csv', sep=',')
df15['weekend'] = False
df16 = pd.read_csv('datasets/paris_weekends.csv', sep=',')
df16['weekend'] = True
paris = pd.concat([df15, df16], ignore_index=True, sort=False)
paris['city'] = 'Paris'

df17 = pd.read_csv('datasets/rome_weekdays.csv', sep=',')
df17['weekend'] = False
df18 = pd.read_csv('datasets/rome_weekends.csv', sep=',')
df18['weekend'] = True
rome = pd.concat([df17, df18], ignore_index=True, sort=False)
rome['city'] = 'Rome'

df19 = pd.read_csv('datasets/vienna_weekdays.csv', sep=',')
df19['weekend'] = False
df20 = pd.read_csv('datasets/vienna_weekends.csv', sep=',')
df20['weekend'] = True
vienna = pd.concat([df19, df20], ignore_index=True, sort=False)
vienna['city'] = 'Vienna'

all_cities = pd.concat([amsterdam, athens, barcelona, berlin, budapest, lisbon, london, paris, rome, vienna], ignore_index=True, sort=False)
del df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20