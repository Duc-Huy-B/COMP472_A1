from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import matplotlib.pyplot as plt



def one_hot_col(data_set, col_name):
    data_set = pd.concat([data_set, pd.get_dummies(data_set[col_name])], axis=1).drop(col_name,axis=1)
    return data_set

def categories(data_set, col_name):
    data_set[col_name] = pd.Categorical(data_set[col_name])
    data_set[col_name] = data_set.cat.codes
    return data_set

def fix_print(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 3000)
    print(x)
    
def plot_percent(file_name,column_name):
    pf = pd.read_csv(file_name)
    ax = pf[column_name].value_counts(normalize=True, sort=False).mul(100).plot(kind='bar', title='Frequency Percentage - '+column_name, xlabel=column_name, ylabel='%', legend='True')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.margins(y=0.1)    
    plt.xticks(rotation=0)
    plt.show()

def base_dt_model(file_name, col_name, ave_a, ave_ma, ave_wa):
    pf = pd.read_csv(file_name)
    #splitting
    x = pf.drop(col_name, axis=1)
    y = pf[col_name]
    #converting
    dtypes = x.dtypes.to_dict()
    for col, typ in dtypes.items():
        if (typ != float):
            x = one_hot_col(x, col)
    if (file_name == "penguins.csv"):
        f = open("penguin-performance.txt", "a")
    else :
        f = open("abalone-performance.txt", "a")
    print("\n--------------------------------------Base-Decision Tree-----------------------------------------", file=f)
    for i in range(5):
        #train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        base_dt = tree.DecisionTreeClassifier(criterion="entropy")
        base_dt.fit(x_train.values, y_train)
        y_pred = base_dt.predict(x_test.values)
        tree.plot_tree(base_dt)
        #plt.show()
        #results / observations
        print(("Confusion Matrix:\n"), confusion_matrix(y_test,y_pred), file=f)
        print("\n",classification_report(y_test,y_pred), file=f)
        f.flush()
        ave_a.append(accuracy_score(y_test,y_pred))
        ave_ma.append(f1_score(y_test,y_pred, average='macro'))
        ave_wa.append(f1_score(y_test,y_pred, average='weighted'))
    f.close()
    
def top_dt_model(file_name, col_name, ave_a, ave_ma, ave_wa):
    pf = pd.read_csv(file_name)
    criterion = "entropy"
    max_depth=3
    min_samples_split=20
    #splitting
    x = pf.drop(col_name, axis=1)
    y = pf[col_name]
    #converting
    dtypes = x.dtypes.to_dict()
    for col, typ in dtypes.items():
        if (typ != float):
            x = one_hot_col(x, col)
    if (file_name == "penguins.csv"):
        f = open("penguin-performance.txt", "a")
    else :
        f = open("abalone-performance.txt", "a")            
    print("--------------------------------------Top-Decision Tree-----------------------------------------", file=f)
    print("\n------------- criterion: %s ------------- max_depth: %s ------------- min_samples_split: %s -------------" % (criterion, max_depth, min_samples_split), file=f)
    for i in range(5):
        #train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        top_dt = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)    
        top_dt.fit(x_train.values, y_train)
        y_pred = top_dt.predict(x_test.values)
        #tree.plot_tree(top_dt)
        #plt.show()
        #results / observations
        print(("Confusion Matrix:\n"), confusion_matrix(y_test,y_pred), file=f)
        print("\n",classification_report(y_test,y_pred), file=f)
        f.flush()
        ave_a.append(accuracy_score(y_test,y_pred))
        ave_ma.append(f1_score(y_test,y_pred, average='macro'))
        ave_wa.append(f1_score(y_test,y_pred, average='weighted'))
    f.close()
    
def base_mlp_model(file_name, col_name, ave_a, ave_ma, ave_wa):
    pf = pd.read_csv(file_name)
    #splitting
    x = pf.drop(col_name, axis=1)
    y = pf[col_name]
    #converting
    dtypes = x.dtypes.to_dict()
    for col, typ in dtypes.items():
        if (typ != float):
            x = one_hot_col(x, col)
    if (file_name == "penguins.csv"):
        f = open("penguin-performance.txt", "a")
    else :
        f = open("abalone-performance.txt", "a")    
    print("--------------------------------------Base-MLP-----------------------------------------", file=f)
    for i in range(5):
        #train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y)        
        #will warn the user for having weights equal to 0, skewing the resulting precison & F1 results
        base_mlp = MLPClassifier(hidden_layer_sizes=(100,100),activation='logistic',solver='sgd')
        base_mlp.fit(x_train.values, y_train)
        y_pred = base_mlp.predict(x_test.values)
        #results / observations        
        print(("Confusion Matrix:\n"), confusion_matrix(y_test,y_pred), file=f)
        print("\n",classification_report(y_test,y_pred), file=f)
        f.flush()
        ave_a.append(accuracy_score(y_test,y_pred))
        ave_ma.append(f1_score(y_test,y_pred, average='macro'))
        ave_wa.append(f1_score(y_test,y_pred, average='weighted'))
    f.close()
    
def top_mlp_model(file_name, col_name, ave_a, ave_ma, ave_wa):
    pf = pd.read_csv(file_name)
    a_function = 'logistic'
    h_layer_size = (10,10,10)
    solver = 'sgd'
    #splitting
    x = pf.drop(col_name, axis=1)
    y = pf[col_name]
    #converting
    dtypes = x.dtypes.to_dict()
    for col, typ in dtypes.items():
        if (typ != float):
            x = one_hot_col(x, col)
    if (file_name == "penguins.csv"):
        f = open("penguin-performance.txt", "a")
    else :
        f = open("abalone-performance.txt", "a") 
    print("--------------------------------------Top-MLP-----------------------------------------", file=f)
    print("\n------------- activation function: %s ------------- hidden layer size: %s ------------- solver: %s -------------" % (a_function, h_layer_size, solver), file=f)
    for i in range(5):
        #train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        #will warn the user for having weights equal to 0, skewing the resulting precison & F1 results
        top_mlp =  MLPClassifier(hidden_layer_sizes=h_layer_size,activation=a_function,solver=solver) 
        top_mlp.fit(x_train.values, y_train)
        y_pred = top_mlp.predict(x_test.values)
        #results / observations 
        print(("Confusion Matrix:\n"), confusion_matrix(y_test,y_pred), file=f)
        print("\n",classification_report(y_test,y_pred), file=f)
        f.flush()
        ave_a.append(accuracy_score(y_test,y_pred))
        ave_ma.append(f1_score(y_test,y_pred, average='macro'))
        ave_wa.append(f1_score(y_test,y_pred, average='weighted'))
    f.close()
        
def ave_cal(file_name, ave_a, ave_ma, ave_wa):
    #calculations
    ave_a_mean = sum(ave_a) / len(ave_a)
    ave_a_var = sum((i - ave_a_mean) ** 2 for i in ave_a) / len(ave_a)
    ave_ma_mean = sum(ave_ma) / len(ave_ma)
    ave_ma_var = sum((i - ave_ma_mean) ** 2 for i in ave_ma) / len(ave_ma)
    ave_wa_mean = sum(ave_wa) / len(ave_wa)
    ave_wa_var = sum((i - ave_wa_mean) ** 2 for i in ave_wa) / len(ave_wa)
    #printing
    if (file_name == "penguins.csv"):
        f = open("penguin-performance.txt", "a")
    else :
        f = open("abalone-performance.txt", "a")
    print("Average Accuracy:", ave_a_mean, file=f)
    print("\n\tAverage Accuracy Variance:", ave_a_var, file=f)
    print("\nAverage Macro-Average:", ave_ma_mean, file=f)
    print("\n\tAverage Macro-Average Variance:", ave_ma_var, file=f)
    print("\nAverage Weighted-Average:", ave_wa_mean, file=f)
    print("\n\tAverage Weighted-Average Variance:", ave_wa_var, file=f)
    f.close()
    ave_a.clear()
    ave_ma.clear()
    ave_wa.clear()
        

def main():
    file_name = "abalone.csv"
    col_name = 'Type'
    ave_a = []
    ave_ma = []
    ave_wa = []
    
    base_dt_model(file_name, col_name, ave_a, ave_ma, ave_wa)
    ave_cal(file_name,ave_a,ave_ma,ave_wa)
    top_dt_model(file_name,col_name, ave_a, ave_ma, ave_wa)
    ave_cal(file_name,ave_a,ave_ma,ave_wa)    
    base_mlp_model(file_name, col_name, ave_a, ave_ma, ave_wa)
    ave_cal(file_name,ave_a,ave_ma,ave_wa)        
    top_mlp_model(file_name, col_name,ave_a,ave_ma,ave_wa)
    ave_cal(file_name,ave_a,ave_ma,ave_wa)
    #plot_percent(file_name, col_name)

if __name__ == '__main__':
    main()