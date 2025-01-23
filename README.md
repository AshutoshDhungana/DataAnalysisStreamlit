# Data Analysis Platform

A web-based platform for data analysis and visualization built with Django and Streamlit.

## Features
- Upload and manage datasets
- Create interactive visualizations (bar plots, scatter plots, line plots)
- User authentication and dataset management
- Interactive UI for data analysis

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
DEBUG=True
SECRET_KEY=your-secret-key
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_HOST=localhost
DB_PORT=5432
```

4. Initialize the database:
```bash
python manage.py migrate
```

5. Run the development server:
```bash
python manage.py runserver  # For Django backend
streamlit run streamlit_app.py  # For Streamlit frontend
```

## Project Structure
- `backend/` - Django backend application
- `frontend/` - Streamlit frontend application
- `manage.py` - Django management script
- `requirements.txt` - Project dependencies 