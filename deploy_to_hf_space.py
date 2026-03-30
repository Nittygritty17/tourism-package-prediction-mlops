import os

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None or HF_TOKEN.strip() == "":
    raise ValueError("HF_TOKEN is missing.")
import os
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_ID = "nittygritty2106/travelpredictionmlops-app"
HF_MODEL_REPO_ID = "nittygritty2106/travelpredictionmlops"

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=SPACE_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

# Content of app.py to be written to a file within the Space
app_py_content_to_write = '\nimport streamlit as st\nimport pandas as pd\nimport joblib\nfrom huggingface_hub import hf_hub_download\n\nst.set_page_config(page_title="Tourism Package Prediction", layout="centered")\n\nREPO_ID = "{{HF_MODEL_REPO_ID}}" # This will be formatted by the deploy script itself\nFILENAME = "model.joblib"\n\n@st.cache_resource\ndef load_model():\n    model_path = hf_hub_download(\n        repo_id=REPO_ID,\n        filename=FILENAME,\n        repo_type="model"\n    )\n    model = joblib.load(model_path)\n    return model\n\nmodel = load_model()\n\nst.title("Tourism Package Purchase Prediction")\nst.write("Enter customer details to predict whether the customer will purchase the tourism package.")\n\nwith st.form("prediction_form"):\n    Age = st.number_input("Age", min_value=18, max_value=100, value=30)\n    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])\n    CityTier = st.selectbox("City Tier", [1, 2, 3])\n    DurationOfPitch = st.number_input("Duration Of Pitch", min_value=1, max_value=100, value=15)\n    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])\n    Gender = st.selectbox("Gender", ["Male", "Female"])\n    NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)\n    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)\n    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])\n    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])\n    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])\n    NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=20, value=2)\n    Passport = st.selectbox("Passport", [0, 1])\n    PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5])\n    OwnCar = st.selectbox("Own Car", [0, 1])\n    NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)\n    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])\n    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=25000)\n\n    submitted = st.form_submit_button("Predict")\n\nif submitted:\n    input_df = pd.DataFrame([{{\n        "Age": Age,\n        "TypeofContact": TypeofContact,\n        "CityTier": CityTier,\n        "DurationOfPitch": DurationOfPitch,\n        "Occupation": Occupation,\n        "Gender": Gender,\n        "NumberOfPersonVisiting": NumberOfPersonVisiting,\n        "NumberOfFollowups": NumberOfFollowups,\n        "ProductPitched": ProductPitched,\n        "PreferredPropertyStar": PreferredPropertyStar,\n        "MaritalStatus": MaritalStatus,\n        "NumberOfTrips": NumberOfTrips,\n        "Passport": Passport,\n        "PitchSatisfactionScore": PitchSatisfactionScore,\n        "OwnCar": OwnCar,\n        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,\n        "Designation": Designation,\n        "MonthlyIncome": MonthlyIncome\n    }}])\n\n    prediction = model.predict(input_df)[0]\n\n    if hasattr(model, "predict_proba"):\n        probability = model.predict_proba(input_df)[0][1]\n        st.write(f"### Probability of Purchase: {{probability:.2%}}")\n\n    if prediction == 1:\n        st.success("The customer is likely to purchase the tourism package.")\n    else:\n        st.error("The customer is unlikely to purchase the tourism package.")\n'

with open("app.py", "w") as f:
    # Format the app_py_content_to_write string with the actual HF_MODEL_REPO_ID
    f.write(app_py_content_to_write.format(HF_MODEL_REPO_ID=HF_MODEL_REPO_ID))

# Content of requirements.txt to be written to a file within the Space
requirements_txt_content_to_write = '\nstreamlit\npandas\njoblib\nhuggingface_hub\nscikit-learn\n'

with open("requirements.txt", "w") as f:
    f.write(requirements_txt_content_to_write)

# Content of Dockerfile to be written to a file within the Space
dockerfile_content_to_write = '\nFROM python:3.9-slim\n\nWORKDIR /app\n\nCOPY requirements.txt ./requirements.txt\nRUN pip install --no-cache-dir -r requirements.txt\n\nCOPY app.py .\n\nEXPOSE 8501\n\nENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]\n'

with open("Dockerfile", "w") as f:
    f.write(dockerfile_content_to_write)

api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=SPACE_ID,
    repo_type="space",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=SPACE_ID,
    repo_type="space",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj="Dockerfile",
    path_in_repo="Dockerfile",
    repo_id=SPACE_ID,
    repo_type="space",
    token=HF_TOKEN
)

print(f"Deployment files pushed to HF Space: https://huggingface.co/spaces/{SPACE_ID}")
