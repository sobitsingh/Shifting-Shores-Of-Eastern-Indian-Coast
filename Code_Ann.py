
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing


# In[2]:


variables = pd.read_csv(r"C:\Users\Sobhit Singh\Desktop\new1.csv")
variables.dtypes


# In[3]:


variables
data = variables.fillna(method='ffill')
data


# In[4]:


df_x=variables.iloc[:,1:10]
df_x


# In[5]:


df_x.isnull().any()
df_x = df_x.fillna(method='ffill')
print(len(df_x))
df_x.Month[1]=float(df_x.Month[1])
print(df_x.Month[1])


# In[6]:


df_x.Month = df_x.Month.astype(float)
df_x.Year = df_x.Year.astype(float)


# In[7]:


df_y=variables.iloc[:,10]
df_y


# In[8]:


print(df_x.iloc[0,:])
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_x)
df_x = pd.DataFrame(scaled_df)

df_x


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf


# In[ ]:


scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(principalDf)
df_x = pd.DataFrame(scaled_df)
df_x


# In[9]:




x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train


# In[ ]:


P, D, Q = np.linalg.svd(df_x, full_matrices=False)
df_x = np.matmul(np.matmul(P, np.diag(D)), Q)


# In[23]:


x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_train)) / np.std(x_train)


# In[16]:



nn=MLPClassifier(activation='tanh',hidden_layer_sizes=(300,200,4),random_state=2,solver = 'adam',alpha = 0.00000001)


# In[ ]:


nn.fit(x_train,y_train)
pred=nn.predict(x_test)


# In[ ]:


print('Accuracy on the training subset: {:.3f}'.format(nn.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(nn.score(x_test, y_test)))


# In[44]:



pq = nn.coefs_[0]
pq


# In[ ]:


import pandas
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# In[ ]:


from sklearn.svm import SVC 
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)

print('The accuracy on the training subset: {:.3f}'.format(svm.score(x_train, y_train)))
print('The accuracy on the test subset: {:.3f}'.format(svm.score(x_test, y_test)))


# In[ ]:


var = pd.read_csv(r"C:\Users\Sobhit Singh\Desktop\futureda.csv")
var


# In[ ]:



x_pred=abs(var.iloc[:,1:10])

x_pred


# In[ ]:



# Fit your data on the scaler object
scaled_df = scaler.fit_transform(x_pred)
x_pred = pd.DataFrame(scaled_df)
a=nn.predict(x_pred)


# In[ ]:


a


# In[ ]:


df = pd.read_csv(r"C:\Users\Sobhit Singh\Desktop\final_data.csv")
print(len(df.Month))


# In[ ]:



x = df.date_for_mean.iloc
y = df.mean_shift


# In[ ]:


plt.plot(year,df.mean_shift)


# In[ ]:


import xlsxwriter

workbook = xlsxwriter.Workbook(r'C:\Users\Sobhit Singh\Desktop\arrays.xlsx')
worksheet = workbook.add_worksheet()



row = 0
col= 9

for data.Shift in (variables):
    worksheet.write_column(row, col,a)
    row+=1
workbook.close()


# In[ ]:



from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split


x_train = scaler.fit(x_train).transform(x_train)
x_test = scaler.fit(x_test).transform(x_test)


svm = SVC()
svm.fit(x_train, y_train)

print('The accuracy on the training subset: {:.3f}'.format(svm.score(x_train, y_train)))
print('The accuracy on the test subset: {:.3f}'.format(svm.score(x_test, y_test)))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(x_train.min(axis=0), 'o', label='Min')
plt.plot(x_train.max(axis=0), 'v', label='Max')
plt.xlabel('Feature Index')
plt.ylabel('Feature Magnitude in Log Scale')
plt.yscale('log')
plt.legend(loc='upper right')


# In[ ]:


plt.plot(df.Year,df.Shift)


# In[ ]:


np.mean(df.Shift)


# In[ ]:


k=0
mean_shift = np.zeros((491,1))
for j in range (1985,2026):
    count = 0
    for i in range(1,12):
        for x in range(0,len(df.Year)):
            if (df.Month[x] == i and  df.Year[x] == j):
                count+=1 
        a = df.Shift[0:count]
        b= np.sum(a)/count 
        mean_shift[k] = b
        k+=1
        
        
        


# In[ ]:


print(df.Shift[0:count])

