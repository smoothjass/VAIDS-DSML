from scipy.stats import norm

data = [20, 25, 30, 42, 110, 111]

mu, std = norm.fit(data)

for i in range(len(data)):
    data[i] = (data[i]-mu)/std

mu, std = norm.fit(data)