import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://faceattendancerealtime-c904f-default-rtdb.firebaseio.com/'
})

ref = db.reference('Students')

data = {
    "111":
        {
            "name": "Salman Aziz Barbhuiya",
            "dept": "CSE",
            "starting_year": 2020,
            "total_attendance": 4,
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "222":
        {
            "name": "Faiyaz Sabab",
            "dept": "CSE",
            "starting_year": 2020,
            "total_attendance": 12,
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "333":
        {
            "name": "Suaib",
            "major": "CSE",
            "starting_year": 2020,
            "total_attendance": 7,
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)