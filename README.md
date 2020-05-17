# Desicion_Tree
## First Step Data Preprocessing:

* The data taken by decision tree is discrete, Bute here there are features
like age and weight which has continuous values .
If I assumed that every value in age for example is a category while
training process,The data will suffer from over fitting which is a big
problem.

So I’d do data binarization.
* The data is binarized in “data_binarization.py”
* The mean is calculated for every feature then every feature value is
compared with mean , if feature val >mean so it takes 1 value else if
feature<mean it takes 0 value
* The new data is saved in a new excel sheet “output.xlsx”

## Second step Implementing ID3 Algorithm:
#### get_entropy(col)
* Takes col and gets its Entropy
#### InfoGain(data,split_attribute_name,target_name="Classs"
* It calculates Total entropy,weighted entropy,Then returns information gain
#### ID3_algorithm(data,features,target="Classs",parent_node_class = None)
* function which calculates the IG for all features getting the best feature for the passed data The function is called recursively then it return Tree
#### Predict(data_test,tree)
* It starts comparing every feature with tree nodes ,so it can predict the output
#### test(data,tree)
* It takes data and tree ,it start to predict the output for every row using “predict ” function.It also prints accuracy
