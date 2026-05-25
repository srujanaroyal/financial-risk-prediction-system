# Financial Risk Classification Project

## Project Structure

- Frontend: Streamlit
- Backend: FastAPI
- ML Model: SVM

## Install Libraries

pip install -r requirements.txt

## Train Model

cd backend
python train_model.py

## Run FastAPI

uvicorn app:app --reload

## Run Streamlit

cd ../frontend
streamlit run streamlit_app.py
