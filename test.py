import joblib
pred=joblib.load("credit.pkl")

# Use the trained classifier to make predictions
res = pred.predict([[100000,1,1,1,1,0,0,0,0,0,12000],[30000.0,2,2,1,38,-1,-1,-1,0,-1,54836.000000],[30000.0,1,2,1,53,0,0,0,0,0,7624.783112]])
print(res)

