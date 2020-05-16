# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:30:29 2020

@author: Bassmala
"""

import pandas as pd  
import numpy as np  
from pprint import pprint  
from sklearn.model_selection import StratifiedShuffleSplit

def get_entropy(target_col):  
    
    elements,counts = np.unique(target_col,return_counts = True)  
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])  
    return entropy  
  
def IG(data,split_attribute_name,target_name="Classs"):  
         
    #Calculate the entropy of the total dataset
    total_entropy = get_entropy(data[target_name])  
          
    #Calculate the values and the corresponding counts for the split attribute   
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)  
      
    #Calculate the weighted entropy  
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*get_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])  
      
    #Calculate the information gain  
    Information_Gain = total_entropy - Weighted_Entropy  
    return Information_Gain  
  
def ID3_algorithm(data,originaldata,features,target_attribute_name="Classs",parent_node_class = None):  
  
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#  
      
    #If all target_values have the same value, return this value  
    if len(np.unique(data[target_attribute_name])) <= 1:  
        return np.unique(data[target_attribute_name])[0]  
      
    #If the dataset is empty, return the mode target feature value in the original dataset  
    elif len(data)==0:  
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]  
      
      
    elif len(features) ==0:  
        return parent_node_class  
      
    #start implementing the tree 
      
    else:  
        #Set the default value for this node --> The mode target feature value of the current node  
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]  
          
        #Select the feature which best splits the dataset  
        item_values = [IG(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset  
        best_feature_index = np.argmax(item_values)  
        best_feature = features[best_feature_index]  
          
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information  
        #gain in the first run  
        tree = {best_feature:{}}  
          
          
        #Remove the feature with the best inforamtion gain from the feature space  
        features = [i for i in features if i != best_feature]  
          
        #Grow a branch under the root node for each possible value of the root node feature  
          
        for value in np.unique(data[best_feature]):  
            value = value  
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets  
            sub_data = data.where(data[best_feature] == value).dropna()  
              
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!  
            subtree = ID3_algorithm(sub_data,dataset,features,target_attribute_name,parent_node_class)  
              
            #Add the sub tree, grown from the sub_dataset to the tree under the root node  
            tree[best_feature][value] = subtree  
              
        return(tree)      
                  



def predict(query,tree,default = 1):  
    
    for key in list(query.keys()):  
        if key in list(tree.keys()):  
            
            try:  
                result = tree[key][query[key]]   
            except:  
                return default  
    
            result = tree[key][query[key]]  
            
            if isinstance(result,dict):  
                return predict(query,result)  
            else:  
                return result  





                
          
  
def test(data,tree):  
    
    queries = data.iloc[:,:-1].to_dict(orient = "records")  
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored  
    predicted = pd.DataFrame(columns=["predicted"])   
    print('predicted',predicted) 
    #Calculate the prediction accuracy 
    true_predicted=0
    data2=[list(row) for row in data.values]  
    for i in range(len(data)):  
       
        predicted.loc[i,"predicted"] = predict(queries[i],tree)
        print(predict(queries[i],tree),"i")
        if data2[i][-1]==predict(queries[i],tree):
            true_predicted=true_predicted+1
            
    accuracy=(true_predicted/float(len(data2)))*100  
     
    print("The accuracy percentage is", accuracy ,"%")   



dataset =  pd.read_excel("output.xlsx",nrows=10000 )
df=dataset
attributes = ["age", "gender", "height", "weight", "ap_hi","ap_low","cholesterol","gluc","smoke","alco","active","Classs"]
df.columns = attributes  
print(df)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(df,df["Classs"]):
    training_data = df.loc[train_index]
    testing_data = df.loc[test_index]  
    
   
tree = ID3_algorithm(training_data,training_data,training_data.columns[:-1])  
pprint(tree)  
print(testing_data,"5swww")
test(testing_data,tree)