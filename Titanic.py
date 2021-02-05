# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %% [markdown]
# The first thing to do is to import and read the dataset into our notebook

# %%
titanic = pd.read_csv("train.csv")
display(titanic.head())

# %% [markdown]
# checking for missing values
# %% [markdown]
# ### Cleaning procces

# %%
print(titanic.isna().sum())
# AGE, CABIN and EMBARKED have some missing values
print(titanic.shape)


# %%
men_age = titanic[titanic["Sex"] == "male"]  # take men apart
# taking the mean age of the mens in the ship
mean_men_age = np.mean(men_age["Age"])
# replacing missing values with the mean
men_age = men_age.fillna({"Age": mean_men_age})
print(men_age["Age"].isna().sum())  # No more missing values

# %% [markdown]
# same logic for women that was used in men

# %%
women_age = titanic[titanic["Sex"] == "female"]
mean_age_women = np.mean(women_age["Age"])
women_age = women_age.fillna({"Age": mean_age_women})
print(women_age["Age"].isna().sum())


# %%
titanic_ok = pd.concat([men_age, women_age])
titanic_ok.sort_values("PassengerId")

# We can confirmn that we havent losed any important information


# %%
assert titanic_ok["Age"].isna().sum() == 0

# %% [markdown]
# For the models we wont be ussing this columns

# %%
col_to_del = ["PassengerId", "Name", "Cabin", "Ticket"]
for col in col_to_del:
    del(titanic_ok[col])

# %% [markdown]
# for the two missing values in "Embarked", I am going to replace with the most common port of embarkation

# %%
most_common_port = titanic_ok["Embarked"].mode()[0]  # Port S, Southampton
titanic_ok = titanic_ok.fillna({"Embarked": most_common_port})
# print(titanic_ok.isna().sum()) #not missing values


# %%
assert titanic_ok["Embarked"].isna().sum() == 0


# %%
titanic_ok

# %% [markdown]
# ### EDA (Exploratoy Data Analysis)

# %%
sns.set_style("whitegrid")
plt.style.use("ggplot")


# %%
sns.catplot(x="Sex", data=titanic_ok, kind="count",
            hue="Survived", col="Pclass", col_wrap=2)
plt.show()


# %%
titanic_ok.groupby(["Pclass", "Sex"])["Survived"].sum()

# %% [markdown]
# As is seeing above, I can notice that the Pclass with the most survivers is the first class, followed by the third and finally the second. this is not a surprise, the priority is to save the "rich" people but what do is surprised is that I found more survivors on the third class than in the second class.
#
# The Famale sex has more chances to survive.

# %%


def to_int(x): return int(x)


titanic_ok["Age"] = titanic_ok["Age"].apply(to_int)


# %%
sns.catplot(x="Sex", y="Age", data=titanic_ok,
            kind="box", sym="", hue="Survived")
sns.swarmplot(x="Sex", y="Age", data=titanic_ok,
              hue="Survived", size=4, dodge=True, color=".2")
plt.show()

# %% [markdown]
# Above I can see how the age is distributed through sex and its hue by either survived or not.
# interesting things to notice:
#       1. The oldest person to survived was a male of approximately 80 years old.
#       2. the majority of the survivors are women between 20 and 35 years old.
#
# %% [markdown]
# ### Modelling.
#

# %%
# Final date set to model.
titanic_ok


# %%
copy_titanic = titanic_ok.copy()


# %%
def sex_dum(sex):
    if sex == "male":
        return 1
    else:
        return 0


copy_titanic["Sex"] = copy_titanic["Sex"].apply(sex_dum)


# %%
def embarked_dum(type):
    if type == "S":
        return 0
    elif type == "Q":
        return 1
    else:
        return 2


copy_titanic["Embarked"] = copy_titanic["Embarked"].apply(embarked_dum)


# %%
copy_titanic  # Ready to modelling


# %%
# labels - Target variables
y = copy_titanic["Survived"].values
# Features
X = copy_titanic.drop("Survived", axis=1).values


# %%
# lets bring the new data set
test_titanic = pd.read_csv("test.csv")
to_predict = test_titanic[["Pclass", "Sex",
                           "Age", "SibSp", "Parch", "Fare", "Embarked"]]
to_predict = to_predict.fillna(method="pad")
assert to_predict["Age"].isna().sum() == 0

to_predict["Age"] = to_predict["Age"].apply(to_int)
to_predict["Sex"] = to_predict["Sex"].apply(sex_dum)
to_predict["Embarked"] = to_predict["Embarked"].apply(embarked_dum)

to_predict  # READT TO USE
# On this cell, I organized the data and left it ready to use. I just used methods previously defined

# %% [markdown]
# #### 1. KNN (K - nearest neigthbors)

# %%


# %%
SEED = 42


# %%
X_scale = scale(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scale, y, test_size=0.3, random_state=SEED, stratify=y)


# %%
k_range = range(1, 30)
scores = list()
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
plt.plot(k_range, scores, marker="o")
plt.ylabel("acurracy")
plt.xlabel("N#_neighbors")
plt.show()

# %% [markdown]
# 8 is the best number of neighbors for this case lets use it

# %%
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(accuracy_score(y_test, y_pred_knn))


# %%
X_to_predict = scale(to_predict.values)
predictions_knn = knn.predict(X_to_predict)
to_report_knn = pd.DataFrame(
    {'PassengerId': test_titanic.PassengerId, 'Survived': predictions_knn})
#to_report_knn.to_csv('submission.csv', index=False)

# %% [markdown]
#  FIRST SUBMISSION IN KEGGLE = 0.75598
# %% [markdown]
# #### 2. Logisctic Regression with hyperparameter tunning

# %%


# %%
log_reg = LogisticRegression()

params_log = {"C": np.logspace(-5, 30)}
grid_log = GridSearchCV(
    estimator=log_reg, param_grid=params_log, cv=10, n_jobs=-1)
grid_log.fit(X_train, y_train)
C_log = grid_log.best_estimator_  # Best parameter


# %%
y_pred_log = grid_log.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_pred_log))


# %%
predictions_log = grid_log.best_estimator_.predict(X_to_predict)
to_report_log = pd.DataFrame(
    {'PassengerId': test_titanic.PassengerId, 'Survived': predictions_log})
#to_report_log.to_csv('submission2.csv', index=False)

# %% [markdown]
# SECOND SUBMISSION IN KEGGLE = 0.76794, this is a better model than the KNN model
# %% [markdown]
# #### 3. Decision tree

# %%


# %%
d_tree = DecisionTreeClassifier(max_depth=6, random_state=SEED)
d_tree.fit(X_train, y_train)
y_pred_dtree = d_tree.predict(X_test)
print(accuracy_score(y_test, y_pred_dtree))


# %%
predictions_dtree = d_tree.predict(X_to_predict)
to_report_dtree = pd.DataFrame(
    {'PassengerId': test_titanic.PassengerId, 'Survived': predictions_dtree})
#to_report_dtree.to_csv('submission3.csv', index=False)

# %% [markdown]
# THIRD SUBMISSION IN KEGGLE = 0.7703
# %% [markdown]
# #### 4. Random Forest

# %%


# %%
r_forest = RandomForestClassifier(
    n_estimators=100, max_depth=4, random_state=SEED)
r_forest.fit(X_train, y_train)
y_pred_rforest = r_forest.predict(X_test)
print(accuracy_score(y_test, y_pred_rforest))


# %%
predictions_rforest = r_forest.predict(X_to_predict)
to_report_rforest = pd.DataFrame(
    {'PassengerId': test_titanic.PassengerId, 'Survived': predictions_rforest})
to_report_rforest.to_csv('submission4.csv', index=False)

# %% [markdown]
# FOURTH SUBMISSION IN KEGGLE = 0.77990
# %% [markdown]
# # To conclude, I must keep working on my models and improve the ways to aboard the problem

# %%
to_predict["Survived"] = to_report_rforest["Survived"]


# %%
to_predict


# %%
#to_predict.to_csv("Final_titanic.csv",index = False)

# %% [markdown]
#

# %%
