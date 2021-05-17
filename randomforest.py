import _pickle as pickle

with open("clustered_profiles.pkl",'rb') as fp:
    cluster_df = pickle.load(fp)


X = cluster_df[['Genre', 'Age', 'Annual_Income', 'Spending_Score']]
y = cluster_df.Cluster


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)

pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)

# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()
