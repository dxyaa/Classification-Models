import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Diya\Downloads\Titanic.csv")

df = df[['Survived', 'Age', 'Sex', 'Pclass']]

df = pd.get_dummies(df, columns=['Sex', 'Pclass'])

df.dropna(inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

example_person = pd.DataFrame({
    'Age': [10],           
    'Sex_female': [0],     
    'Sex_male': [1],
    'Pclass_1': [0],      
    'Pclass_2': [0],       
    'Pclass_3': [1]        
})


example_person_scaled = scaler.transform(example_person)

survival_prediction = knn_model.predict(example_person_scaled)

if survival_prediction[0] == 0:
    print("The example person is predicted to perish.")
else:
    print("The example person is predicted to survive.")