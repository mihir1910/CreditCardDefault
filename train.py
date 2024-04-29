import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle,joblib
df=pd.read_csv("credit_data.csv")
# print(df)
df.drop('Unnamed: 0',axis=1,inplace=True)

for col in df.select_dtypes(include=['int', 'float']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer') if df[col].dtype == 'int' else pd.to_numeric(df[col], downcast='float')
print(df.info())
X=df.iloc[:, :-1]
y=df.iloc[:,-1]
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)

scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')
print("Cross-validated scores:", scores)
print("Mean CV score:", scores.mean())
print("Standard deviation of CV scores:", scores.std())

joblib.dump(rf_clf,'credit.pkl')