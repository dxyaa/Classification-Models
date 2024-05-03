import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
df = pd.read_csv(r"C:\Users\Diya\Downloads\Titanic.csv")
df = df[['Survived', 'Age', 'Sex', 'Pclass']]
df = pd.get_dummies(df, columns=['Sex', 'Pclass'])
df.dropna(inplace=True)
df.head()
x = df.drop('Survived', axis=1)
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)
model.score(x_test, y_test)
cross_val_score(model, x, y, cv=5).mean()
y_predicted = model.predict(x_test)
disp1=ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['Perished', 'Survived'], cmap='Blues', xticks_rotation='vertical')
plt.show()

example = pd.DataFrame({
    'Age': [30],           
    'Sex_female': [0],     
    'Sex_male': [1],
    'Pclass_1': [0],      
    'Pclass_2': [1],       
    'Pclass_3': [0]        
})

survival_probability = model.predict_proba(example)
perishing_percentage = survival_probability[0][0] * 100
surviving_percentage = survival_probability[0][1] * 100

print("Perishing Probability: {:.2f}%".format(perishing_percentage))
print("Surviving Probability: {:.2f}%".format(surviving_percentage))