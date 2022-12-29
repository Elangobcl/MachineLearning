##Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV,RandomizedSearchCV #train test split library
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline # performs pipeline of activities
from sklearn.cluster import kMeans
#Text Extraction, takes unique words in complete dataset and finds occurance of unique words in each row/sentence
from sklearn.feature_extraction.text import CountVectorizer
#RandomizedSearchCV - if computation power is contrain, runs grid search for n_iter iterations only
#from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# Scaling of input data into -1 to 1 
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
#PCA dimensionality reduction
from sklearn.decomposition import PCA

##Drop
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

##Segreate Input and Outut from dataset into individual DataFrame
inputs = df.drop('Survived',axis='columns')
target = df.Survived ## target/output Should be in single dimension/coloumn

##1 hot Encoding (converting categorical values/NaN data to numerical data)
dummies = pd.get_dummies(inputs.Sex) ## M/F caltegorical & NaN data
pd.get_dummies(inputs.Sex, drop_first=True) # Drops 1st coloumn to avoid dummy varaible trap
inputs = pd.concat([inputs,dummies],axis='columns') ## concat input & EncodedData (dummies)
inputs.drop(['Sex','male'],axis='columns',inplace=True) ## drop categorical data (sex) & one of the EncodedDatas (M/F), to avoid dummy varaible trap

## Check any I/P cloumns has Nan data
inputs.columns[inputs.isna().any()] #returns coloumn (Age) which has Nan data
inputs.Age = inputs.Age.fillna(inputs.Age.mean()) ##fill Nan data with mean data
df.isnull().sum() #Total no.of null data in each feature/coloumn

## Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)

##Load algorithm Train data and accuracy
model = GaussianNB()
model.fit(X_train,y_train) #Train
model.score(X_test,y_test) #Accuracy
model.predict(X_test[0:10]) #Predict

##Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()), #API Transforms input data into  table of unique words
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

##GridSearchCV
svm = SVC()
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],                         #Different parameters for HyperTUning
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False) # cv = 5, 5 folds since folds train_test_split not required

dir(clf) #returns Methods available in the function

##RandomizedSearchCV, runs grid search for 2 iterations only
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=2
)
rs.fit(iris.data, iris.target)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']] #converts results into DataFrame (rows and coloumn format)

##L1 Regularization for Linear regression models to handle Overfit (train_score > test_score)
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)

##L2 Regularization for Linear regression models to handle Overfit (train_score > test_score)
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)

##Scaling Input data (transforming data in the range -1 to 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_Input)
X_scaled

##PCA in %
pca = PCA(0.95) #retaining 95% of principal Component/Input feature
X_pca = pca.fit_transform(X_Input)
X_pca.shape

##PCA in I/P feature (n_components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_Input)
X_pca.shape

## K-Fold Cross Validation
kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)
    """
    output
    [3 4 5 6 7 8] [0 1 2]
    [0 1 2 6 7 8] [3 4 5]
    [0 1 2 3 4 5] [6 7 8]

    """
kf_stratified = StratifiedKFold(n_splits=3) ## uniformly shares categories data (M/F, Day/Night pics) to train (training) & test (validation)
for train_index, test_index in kf_stratified.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

folds = StratifiedKFold(n_splits=3)
for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]

                #model(),InputFeature,Target
cross_val_score(DecisionTreeClassifier(), X_Input, y_Target, cv=5)

##Bagging
bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(), #model
    n_estimators=100, #Subset
    max_samples=0.8, #train_dataSet
    oob_score=True, #out_of_box(oob) - if training dataset is 80%, the missed data (in the 80%) will be automatically taken for training
    random_state=0
)
bag_model.fit(X_train, y_train)
bag_model.oob_score_ # score out of missed data (considered as test data) in the train dataset (80%)

##Function to handle multiple model
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

##KMeans Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

##Scale MinMax data
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

##Elbow Plot
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

