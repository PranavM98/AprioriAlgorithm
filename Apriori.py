import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori


# Importing the dataset
dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
type(dataset)



for col in dataset.columns:
    print (col)


df=dataset.drop(dataset[dataset.Admit<0.65].index)
df=df.drop(['Admit'],axis=1)



for index,rows in df.iterrows():
    if df.loc[index,'Research']==1:
        df.loc[index,'Research']='Research-1'
    else:
        df.loc[index,'Research']='Research-0'




for index, row in df.iterrows():

    if df.loc[index,'GRE_Score'] >=320:
        df.loc[index,'GRE_Score']='GRE-1'
    elif df.loc[index,'GRE_Score']<300:
        df.loc[index,'GRE_Score']='GRE-3'
    else:
        df.loc[index,'GRE_Score']='GRE-2'



df.groupby('GRE_Score').count()



for index, row in df.iterrows():

    if df.loc[index,'TOEFL_Score'] >110:
        df.loc[index,'TOEFL_Score']='TOEFL-1'
    elif df.loc[index,'TOEFL_Score']<100:
        df.loc[index,'TOEFL_Score']='TOEFL-3'
    else:
        df.loc[index,'TOEFL_Score']='TOEFL-2'

df.groupby('TOEFL_Score').count()


for index, row in df.iterrows():

    if df.loc[index,'SOP']==4.5:
        df.loc[index,'SOP']=5
    elif df.loc[index,'SOP']==3.5:
        df.loc[index,'SOP']=4
    elif df.loc[index,'SOP']==2.5:
        df.loc[index,'SOP']=3
    elif df.loc[index,'SOP']==1.5:
        df.loc[index,'SOP']=2
    elif df.loc[index,'SOP']==0.5:
        df.loc[index,'SOP']=1
    else:
        continue

df.groupby('SOP').count()

for index, row in df.iterrows():

    if df.loc[index,'LOR']==4.5:
        df.loc[index,'LOR']=5
    elif df.loc[index,'LOR']==3.5:
        df.loc[index,'LOR']=4
    elif df.loc[index,'LOR']==2.5:
        df.loc[index,'LOR']=3
    elif df.loc[index,'LOR']==1.5:
        df.loc[index,'LOR']=2
    elif df.loc[index,'LOR']==0.5:
        df.loc[index,'LOR']=1
    else:
        continue

df.groupby('LOR').count()



for index,rows in df.iterrows():
    if df.loc[index,'SOP']==5:
        df.loc[index,'SOP']='SOP-5'

    elif df.loc[index,'SOP']==4:
        df.loc[index,'SOP']='SOP-4'

    elif df.loc[index,'SOP']==3:
        df.loc[index,'SOP']='SOP-3'

    elif df.loc[index,'SOP']==2:
        df.loc[index,'SOP']='SOP-2'

    else:
        df.loc[index,'SOP']='SOP-1'


for index,rows in df.iterrows():
    if df.loc[index,'LOR']==5:
        df.loc[index,'LOR']='LOR-5'

    elif df.loc[index,'LOR']==4:
        df.loc[index,'LOR']='LOR-4'

    elif df.loc[index,'LOR']==3:
        df.loc[index,'LOR']='LOR-3'

    elif df.loc[index,'LOR']==2:
        df.loc[index,'LOR']='LOR-2'

    else:
        df.loc[index,'LOR']='LOR-1'



for index,rows in df.iterrows():
    if df.loc[index,'CGPA']>=6.75 and df.loc[index,'CGPA']<7.25:
        df.loc[index,'CGPA']='CGPA:6.75-7.25'
    elif df.loc[index,'CGPA']>=7.25 and df.loc[index,'CGPA']<7.75:
        df.loc[index,'CGPA']='CGPA:7.25-7.5'
    elif df.loc[index,'CGPA']>=7.75 and df.loc[index,'CGPA']<8.25:
        df.loc[index,'CGPA']='CGPA:7.75-8.25'
    elif df.loc[index,'CGPA']>=8.25 and df.loc[index,'CGPA']<8.75:
        df.loc[index,'CGPA']='CGPA:8.25-8.75'
    elif df.loc[index,'CGPA']>=8.75 and df.loc[index,'CGPA']<9.25:
        df.loc[index,'CGPA']='CPGA:8.75-9.25'
    elif df.loc[index,'CGPA']>=9.25 and df.loc[index,'CGPA']<9.75:
        df.loc[index,'CGPA']='CGPA:9.25-9.75'
    elif df.loc[index,'CGPA']>=9.75:
        df.loc[index,'CGPA']='CGPA:9.75-10.0'
    

df.to_csv('datafinal.csv',index=False)
##################Scatter Plots######################
cgpa= df["CGPA"]
university_ranking = df["University_Rating"]
plt.scatter(cgpa, university_ranking)
plt.title("Scatter plot of CGPA vs University Rating")
plt.xlabel("CGPA")
plt.ylabel("University Ranking")

######################################################



datawith5=df.loc[(df.University_Rating==5)]
datawith5=datawith5.drop(['University_Rating'],axis=1)

datawith4=df.loc[(df.University_Rating==4)]

datawith4=datawith4.drop(['University_Rating'],axis=1)

datawith3=df.loc[(df.University_Rating==3)]

datawith3=datawith3.drop(['University_Rating'],axis=1)


datawith2=df.loc[(df.University_Rating==2)]

datawith2=datawith2.drop(['University_Rating'],axis=1)

datawith1=df.loc[(df.University_Rating==1)]

datawith1=datawith1.drop(['University_Rating'],axis=1)


datawith5.to_csv('data5.csv',index=False)
datawith4.to_csv('data4.csv',index=False)
datawith3.to_csv('data3.csv',index=False)
datawith2.to_csv('data2.csv',index=False)
datawith1.to_csv('data1.csv',index=False)

#######Density Plots#######

'''
p1=sns.kdeplot(datawith1['CGPA'], shade=True, color="black")
p1=sns.kdeplot(datawith2['CGPA'], shade=True, color="b")
p1=sns.kdeplot(datawith3['CGPA'], shade=True, color="y")
p1=sns.kdeplot(datawith4['CGPA'], shade=True, color="g")
p1=sns.kdeplot(datawith5['CGPA'], shade=True, color="r")


p2=sns.kdeplot(datawith1['GRE_Score'], shade=True, color="g")
p2=sns.kdeplot(datawith2['GRE_Score'], shade=True, color="r")
p2=sns.kdeplot(datawith3['GRE_Score'], shade=True, color="black")
p2=sns.kdeplot(datawith4['GRE_Score'], shade=True, color="b")
p2=sns.kdeplot(datawith5['GRE_Score'], shade=True, color="y")

p3=sns.kdeplot(datawith1['LOR'], shade=True, color="g")
p3=sns.kdeplot(datawith2['LOR'], shade=True, color="r")
p3=sns.kdeplot(datawith3['LOR'], shade=True, color="black")
p3=sns.kdeplot(datawith4['LOR'], shade=True, color="b")
p3=sns.kdeplot(datawith5['LOR'], shade=True, color="y")

p4=sns.kdeplot(datawith1['SOP'], shade=True, color="g")
p4=sns.kdeplot(datawith2['SOP'], shade=True, color="r")
p4=sns.kdeplot(datawith3['SOP'], shade=True, color="black")
p4=sns.kdeplot(datawith4['SOP'], shade=True, color="b")
p4=sns.kdeplot(datawith5['SOP'], shade=True, color="y")

'''
#sns.plt.show()
#sns.plt.show()
##########################




#### Apriori Algorithm

data=pd.read_csv('data5and4.csv',header=None)
records = []
for i in range(0, 166):
    records.append([str(data.values[i,j]) for j in range(1,7)])

records=records[1:len(data)]

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df5 = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df5, min_support=0.6, use_colnames=True)
print(frequent_itemsets)
frequent_itemsets=frequent_itemsets.sort_values(by='support',ascending=0)


# ASSOCIATION RULES
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.00002)
rules=rules.sort_values(by="confidence",ascending=0)
rules


