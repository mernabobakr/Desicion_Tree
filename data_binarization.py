import pandas as pd

data= pd.read_csv("cardio_train.csv", nrows=10000,delimiter=";") 

#plot histogram to see data 
#data.hist(bins=8, figsize=(20,15))
#plt.show()
first_column = data.columns[0]
df1 = data.drop([first_column], axis=1)
print(df1)


df1.to_csv('file.csv', index=False)




attributes = ["age", "gender", "height", "weight", "ap_hi","ap_low","cholesterol","gluc","smoke","alco","active","Classs"]
df1.columns = attributes

df = [list(row) for row in df1.values]
#print(df)

#function takes continous data then it turns it to 0 1 output
def binarization(excel_data,data,col_no):
    #get the threshold which is the mean of the feature
    threshold=excel_data[excel_data.columns[col_no]].mean()
    for i in range(len(data)):
        if data[i][col_no]<threshold:
            data[i][col_no]=0
            
        else:
            data[i][col_no]=1
            
    return data        



#data=age_binarization(df,0)

#0 FOR female & 1 for male

def gender_binarization(data,col_no):
    for i in range(len(data)):
        if data[i][col_no]==1:
            data[i][col_no]=0
            
        else:
            data[i][col_no]=1
            
    return data     


def data_binarization(excel_data,data):
    #binarize age values
    data=binarization(excel_data,data,0)
    
    #binarize gender with ) 0,1
    data=gender_binarization(data,1)
    
    #binarize height values
    data=binarization(excel_data,data,2)
    
    #binarize weight values
    data=binarization(excel_data,data,3)
    
    #binarize ap_hi values 
    data=binarization(excel_data,data,4)
    
    #binarize ap_lo values 
    data=binarization(excel_data,data,5)
    
    #binarize chlosterol values 
    data=binarization(excel_data,data,6)
    
    
    #binarize gluc values 
    data=binarization(excel_data,data,7)
    return data
    
    
    
    
    
    
    
binarized_data=data_binarization(df1,df)
pd.DataFrame(binarized_data).to_csv('binarizee.csv', header=False, index=False)   
print(pd)