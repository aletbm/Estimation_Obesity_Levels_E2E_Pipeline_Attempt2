import requests

API_URL = "https://obesity-level-api-761922006747.us-east1.run.app/predict"
#API_URL = "http://localhost:8080/predict"

sample_input = {'gender': 'Male',
                'age': 21.872484,
                'height': 1.699998,
                'overweight_familiar': 'yes',
                'eat_hc_food': 'yes',
                'eat_vegetables': 2.0,
                'main_meals': 2.970675,
                'snack': 'Sometimes',
                'smoke': 'no',
                'drink_water': 2.0,
                'monitoring_calories': 'no',
                'physical_activity': 0.0,
                'use_of_technology': 0.169294,
                'drink_alcohol': 'no',
                'transportation_type': 'Public_Transportation',
                #'obesity_level': 'Obesity_Type_I'
                }


sample_input = {"features": sample_input}

response = requests.post(API_URL, json=sample_input)

print(f"Status code: {response.status_code}")
print("Prediction:", response.json())
