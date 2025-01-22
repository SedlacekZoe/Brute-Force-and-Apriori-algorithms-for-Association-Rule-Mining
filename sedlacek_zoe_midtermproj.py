#!/usr/bin/env python
# coding: utf-8

# In[13]:


#pip install pandas #use if need to install
#pip install mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from itertools import combinations
from time import process_time


# In[14]:


fileName = input('Enter data file name ')
fileCheck = 0
while (not fileCheck):
    try:
        file = open(fileName, "r")
    except: 
        fileName = input('File could not be opened. Enter new file name ')
    else:
        fileCheck = 1


# In[15]:


minSup = float(input('Enter min support value '))


# In[16]:


minCon = float(input('Enter min confidence value '))


# In[17]:


file = open(fileName)
#file = open('projdata1.txt') #For faster testing, comment out
transactions = []
line = file.readline()
#Remove trailing space, split into attributes, add to list
while(line != ''):
    line = line.strip()
    splitLine = line.split(',')
    transactions.append(splitLine)
    line = file.readline()
print("Transactions:")
for t in transactions:
    print(t)
print('\n')


# In[18]:


#use transaction encoder to get one hot dataframe of attributes
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transactionDF = pd.DataFrame(te_ary, columns=te.columns_)


# In[19]:


#Apriori -- use mlxtend implementation to check freq itemsets and association rules
apStart = process_time()
freqItemsets = apriori(transactionDF, min_support=minSup, use_colnames=True)
freqItemsets['length'] = freqItemsets['itemsets'].apply(lambda x: len(x))
apEnd = process_time()


# In[20]:


mlxRuleDf = association_rules(freqItemsets, metric="confidence", min_threshold=minCon)


# In[21]:


#brute force
bfStart = process_time()
#combinations & support
#len(attList) + 1
totalTransactions = len(transactionDF)
attList = list(transactionDF.columns)
supDF = pd.DataFrame()
#For each number of attributes
for numAtt in range(1, len(attList) + 1):
    freqItemBool = False
    #For each combination of number of attributes
    for combo in list(combinations(attList, numAtt)):
        prev = 1
        #Create df for combination by and-ing df of all attributes in it
        for att in combo:
            attDF = transactionDF[att] & prev
            prev = attDF
        #display(combo)
        #Remove rows that do not have all attributes in combo
        comboDF = transactionDF[attDF]
        support = len(comboDF) / len(transactionDF)
        if support >= minSup:
            freqItemBool = True
            row = {'combo': combo, 'support': support, 'length': numAtt}
            supDF = pd.concat([supDF, pd.DataFrame([row])], ignore_index=True)
    #if, after all combos of length X have been checked, none are frequent, brute force breaks
    if freqItemBool == False:
        break
bfEnd = process_time()


# In[22]:


#Check rules
myRuleDF = pd.DataFrame()
for length in range(2, max(supDF['length']) + 1):
    for freqCombo in supDF[supDF['length'] == length]['combo']:
        #print(freqCombo)
        freqComboSup = supDF.loc[supDF['combo'] == freqCombo]['support'].iloc[0]
        #print("Support = " + str(freqComboSup))
        #print("Frequent Subsets:")
        #print('\n')
        for subsetLength in range(1, length):
            for subsetCombo in list(combinations(freqCombo, subsetLength)):
                #print(subsetCombo)
                subsetSup = supDF.loc[supDF['combo'] == subsetCombo]['support'].iloc[0]
                #print("Support = " + str(subsetSup))
                
                #for each freq subset on the lhs, try all freq subsets as possible rhs
                #print("possible rhs:")
                for rhsLength in range(1, length + 1):
                    for rhs in list(combinations(freqCombo, rhsLength)):
                        rhsSup = supDF.loc[supDF['combo'] == rhs]['support'].iloc[0]
                        validRule = True
                        #Remove rhs from consideration if it includes the lhs
                        for subsetAttribute in subsetCombo:
                            if subsetAttribute in rhs:
                                validRule = False
                        if validRule:
                            #print(rhs)
                            totalTup = tuple(sorted(subsetCombo + rhs))
                            totalSup = supDF.loc[supDF['combo'] == totalTup]['support'].iloc[0]
                            if totalSup / subsetSup >= minCon: 
                                ruleRow = {'antecedents': subsetCombo, 'consequents': rhs, 'antecedent support': subsetSup, 'consequent support': rhsSup, 
                                           'support': totalSup,'confidence': totalSup / subsetSup}
                                myRuleDF = pd.concat([myRuleDF, pd.DataFrame([ruleRow])], ignore_index=True)
                                #print("RULE")
                                #print(str(subsetCombo) + "-->" + str(rhs))
                                #print("Confidence = " + str(freqComboSup / subsetSup))
                            
                    #print("\n")
        #print("---------------------")


# In[23]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("Apriori Frequent Itemsets")
print(freqItemsets)
print("\n")
print("Apriori Rules")
print(mlxRuleDf)
print("\n")
print("Brute Force Frequent Itemsets")
print(supDF)
print("\n")
print("Brute Force Rules")
print(myRuleDF.drop_duplicates())
print("\n")


# In[24]:


print("Apriori Time: " + str(apEnd - apStart))
print("Brute Force Time: " + str(bfEnd - bfStart))
print("Apriori took " + str(100 * (apEnd - apStart) / (bfEnd - bfStart)) + " percent as long as Brute Force")


# In[ ]:


print('\n')
input("Press 'Enter' to exit")

