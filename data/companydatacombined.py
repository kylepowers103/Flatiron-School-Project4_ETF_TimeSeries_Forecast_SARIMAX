import pandas as pd

df=pd.read_csv('/Users/powersky/Documents/11FlatIronSchoolFolder/Projects/Project_4/Project_4/data/Nasdaqcompanies.csv')
df=df[['Symbol', 'Name']]
df2=pd.read_csv('/Users/powersky/Documents/11FlatIronSchoolFolder/Projects/Project_4/Project_4/data/S&p500companies.csv')
df2=df2[['Symbol', 'Name']]

frames = [df, df2]
company_list = pd.concat(frames)
print(company_list.head())

company_list.to_csv('companies.csv')
