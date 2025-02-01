import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel





# Example form data (replace with actual form data)
form_data = {
    'gender': 'Male',
    'year_of_study': 'Second year',
    'bac_specialty': 'Science',
    'study_place': "At home",
    'study_preference': 'With a group',
    'learning_methods': 'Interactive_workshops,Mind_mapping_and_visualization',
    'resources_used': 'YouTube_videos,Online_documents',
    'use_planner': 'No',
    'problems': 'Bad quality of internet',
    'second_problem': 'Teachers\' teaching method',
    'third_problem': 'Not having a good (powerful) computer',
    'satisfaction_with_program': 3,
    'external_activities': 'Yes',
    'motivation_for_joining': 7,
    'interest_in_ai': 8,
    'feedback_influence': 'Negatively',
    'sleep_hours': '5-7 hours',
    'stress_handling': 7,
    'english_level': 8,
    'programming_level': 5,
    'cs_level': 7,
    'math_level': 6,
    'study_hours': '20-30'
}

# Mapping dictionaries (same as used during preprocessing)
gender_mappings = {'Female': 0, 'Male': 1}
year_mapping = {'First year': 1, 'Second year': 2, 'Third year': 3, 'Fourth year': 4}
bac_specialty_mapping = {'Mathematics': 1, 'Science': 2, 'Technical maths': 3}
study_place_mapping = {"In your room (dorms)": 1, "In the school's library": 2, "At home": 3, "In the dorms' library": 2}
mappings = {'Alone': 1, 'In peer': 2, 'With a group': 3}
planer_mapping = {'No': 0, 'Yes': 1}
problems_mapping = {'Teaching language': 1, 'Bad quality of internet': 2, 'Lack of time': 3, 'Lack of previous knowledge or experience in the field': 4, 'Teachers\' teaching method': 5, 'Pressure': 6, 'Not having a good (powerful) computer': 7, 'Private life problems': 8}
feedback_mapping = {'You do not care about': 0, 'Positively': 1, 'Negatively': 2}
sleep_mapping = {'5-7 hours': 6, '8 hours or more': 8, '4 hours or less': 2}

# Split study hours into start and end
lat_long = form_data['study_hours'].split("-")
form_data['study_hours_start'] = int(lat_long[0])
form_data['study_hours_end'] = int(lat_long[1])

# Preprocess the form data
preprocessed_data = {
    'gender': gender_mappings[form_data['gender']],
    'year_of_study': year_mapping[form_data['year_of_study']],
    'bac_specialty': bac_specialty_mapping[form_data['bac_specialty']],
    'study_place': study_place_mapping[form_data['study_place']],
    'study_preference': mappings[form_data['study_preference']],
    'use_planner': planer_mapping[form_data['use_planner']],
    'problems': problems_mapping[form_data['problems']],
    'second_problem': problems_mapping[form_data['second_problem']],
    'third_problem': problems_mapping[form_data['third_problem']],
    'satisfaction_with_program': form_data['satisfaction_with_program'],
    'external_activities': planer_mapping[form_data['external_activities']],
    'motivation_for_joining': form_data['motivation_for_joining'],
    'interest_in_ai': form_data['interest_in_ai'],
    'feedback_influence': feedback_mapping[form_data['feedback_influence']],
    'sleep_hours': sleep_mapping[form_data['sleep_hours']],
    'stress_handling': form_data['stress_handling'],
    'english_level': form_data['english_level'],
    'programming_level': form_data['programming_level'],
    'cs_level': form_data['cs_level'],
    'math_level': form_data['math_level'],
    'study_hours_start': form_data['study_hours_start'],
    'study_hours_end': form_data['study_hours_end']
}

# Add learning methods and resources used
learning_methods = ['Asking_friends','Interactive_workshops', 'Learning_by_practicing',
       'Mind_mapping_and_visualization']
resources_used = ['Lectures', 'Books',
       'Online_courses_and_tutorials', 'Project_based_learning', 'ChatGpt',
       'Course_material', 'Mentors', 'Online_documents', 'YouTube_videos',
       ]

for method in learning_methods:
    preprocessed_data[method] = 1 if method in form_data['learning_methods'] else 0

for resource in resources_used:
    preprocessed_data[resource] = 1 if resource in form_data['resources_used'] else 0

# Define the column order
column_order = ['gender', 'year_of_study', 'bac_specialty', 'study_place',
       'study_preference', 'use_planner', 'problems', 'second_problem',
       'third_problem', 'satisfaction_with_program', 'external_activities',
       'motivation_for_joining', 'interest_in_ai', 'feedback_influence',
       'sleep_hours', 'stress_handling', 'english_level', 'programming_level',
       'cs_level', 'math_level', 'study_hours_start', 'study_hours_end',
       'Lectures', 'Books',
       'Online_courses_and_tutorials', 'Project_based_learning', 'ChatGpt',
       'Course_material', 'Mentors', 'Online_documents', 'YouTube_videos',
       'Asking_friends', 'Interactive_workshops', 'Learning_by_practicing',
       'Mind_mapping_and_visualization']

# Convert to DataFrame with specified column order
form_df = pd.DataFrame([preprocessed_data], columns=column_order)

# Predict using the best decision tree model
best_dt = joblib.load('best_decision_tree_model.pkl')

prediction = best_dt.predict(form_df)

print("Predicted 1Y_avg:", prediction[0])
app = FastAPI()

class FormData(BaseModel):
    gender: str
    year_of_study: str
    bac_specialty: str
    study_place: str
    study_preference: str
    learning_methods: str
    resources_used: str
    use_planner: str
    problems: str
    second_problem: str
    third_problem: str
    satisfaction_with_program: int
    external_activities: str
    motivation_for_joining: int
    interest_in_ai: int
    feedback_influence: str
    sleep_hours: str
    stress_handling: int
    english_level: int
    programming_level: int
    cs_level: int
    math_level: int
    study_hours: str

@app.post("/predict")
def predict(form_data: FormData):
    try:
        # Split study hours into start and end
        lat_long = form_data.study_hours.split("-")
        form_data.study_hours_start = int(lat_long[0])
        form_data.study_hours_end = int(lat_long[1])

        # Preprocess the form data
        preprocessed_data = {
            'gender': gender_mappings[form_data.gender],
            'year_of_study': year_mapping[form_data.year_of_study],
            'bac_specialty': bac_specialty_mapping[form_data.bac_specialty],
            'study_place': study_place_mapping[form_data.study_place],
            'study_preference': mappings[form_data.study_preference],
            'use_planner': planer_mapping[form_data.use_planner],
            'problems': problems_mapping[form_data.problems],
            'second_problem': problems_mapping[form_data.second_problem],
            'third_problem': problems_mapping[form_data.third_problem],
            'satisfaction_with_program': form_data.satisfaction_with_program,
            'external_activities': planer_mapping[form_data.external_activities],
            'motivation_for_joining': form_data.motivation_for_joining,
            'interest_in_ai': form_data.interest_in_ai,
            'feedback_influence': feedback_mapping[form_data.feedback_influence],
            'sleep_hours': sleep_mapping[form_data.sleep_hours],
            'stress_handling': form_data.stress_handling,
            'english_level': form_data.english_level,
            'programming_level': form_data.programming_level,
            'cs_level': form_data.cs_level,
            'math_level': form_data.math_level,
            'study_hours_start': form_data.study_hours_start,
            'study_hours_end': form_data.study_hours_end
        }

        # Add learning methods and resources used
        for method in learning_methods:
            preprocessed_data[method] = 1 if method in form_data.learning_methods else 0

        for resource in resources_used:
            preprocessed_data[resource] = 1 if resource in form_data.resources_used else 0

        # Convert to DataFrame with specified column order
        form_df = pd.DataFrame([preprocessed_data], columns=column_order)

        # Predict using the best decision tree model
        prediction = best_dt.predict(form_df)

        if prediction[0] == 1:
            return {"Predicted 1Y_avg": "Low"}
        elif prediction[0] == 2:
            return {"Predicted 1Y_avg": "Moderate"}
        elif prediction[0] == 3:
            return {"Predicted 1Y_avg": "High"}
        elif prediction[0] == 4:
            return {"Predicted 1Y_avg": "Very High"}
        else:
            return {"Predicted 1Y_avg": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))