import pandas as pd

chunksize = 10 ** 6
df = pd.DataFrame()
for chunk in pd.read_csv('itineraries.csv', chunksize=chunksize, sep=","):
    print("processing chunk")
    # chunk is a DataFrame. To "process" the rows in the chunk:
    for index, row in chunk.iterrows():
        if row['startingAirport'] == 'VIE':
            df = df.append(row)
#https://stackoverflow.com/questions/25962114/how-do-i-read-a-large-csv-file-with-pandas

df.to_csv('itineraries-from-VIE.csv', index=False)