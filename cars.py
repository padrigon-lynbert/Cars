from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import cross_validate

from connections import get_df

df = get_df()

X = df.drop({'Car_ID', 'Owner_Type'}, axis=1)
y = df['Owner_Type']


numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessing = ColumnTransformer([
    ('scaler', StandardScaler(), numerical_features),
    ('one-hot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


logistic = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

ensemble = VotingClassifier([
    ('logistic', logistic),
    ('rf', rf),
    ('xgb', xgb)
], voting='hard')

pipeline_voting = Pipeline([
    ('preprocessor', preprocessing),
    ('ensemble', ensemble)
])


scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
classification_result_ensemble = cross_validate(pipeline_voting, X, y, scoring=scoring)

mean_voting = {key: value.mean() for key, value in classification_result_ensemble.items()}

for metric, mean in mean_voting.items():
    print(f'{metric}: {mean:.4f}')