import streamlit as st
import pandas as pd
import requests
import json
from PIL import Image
import io
import base64
import numpy as np

# Configure the app
st.set_page_config(
    page_title="Data Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

# Initialize session state variables
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

def get_session():
    return st.session_state.auth_token

def refresh_page():
    st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1
    st.rerun()

def landing_page():
    st.title("📊 Welcome to Data Analysis Platform")
    
    # Hero section
    st.markdown("""
    ### Transform Your Data into Insights
    Upload your datasets and create beautiful visualizations with just a few clicks.
    No coding required!
    """)
    
    # Features section
    st.markdown("---")
    st.header("✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📈 Rich Visualizations
        - Bar plots
        - Line plots
        - Scatter plots
        - Histograms
        - Box plots
        - Violin plots
        - Pie charts
        - And more!
        """)
    
    with col2:
        st.markdown("""
        #### 🔒 Secure & Private
        - User authentication
        - Private datasets
        - Secure data storage
        - Data encryption
        """)
    
    with col3:
        st.markdown("""
        #### 🚀 Easy to Use
        - Intuitive interface
        - No coding required
        - Interactive plots
        - Real-time updates
        """)
    
    # Call to action
    st.markdown("---")
    st.markdown("### Ready to Start?")
    
    col1, col2, _ = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Login"):
            st.session_state.page = "login"
            st.rerun()
    
    with col2:
        if st.button("Learn More"):
            st.markdown("""
            ### How It Works
            1. **Upload** your CSV or Excel files
            2. **Select** the type of visualization
            3. **Customize** your plots
            4. **Share** your insights
            
            ### Supported File Types
            - CSV files (.csv)
            - Excel files (.xlsx)
            
            ### Need Help?
            Contact our support team for assistance.
            """)

def login_page():
    st.header("Login")
    
    # Add tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    # Login tab
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/auth/login/",
                        json={"username": username, "password": password}
                    )
                    
                    if response.status_code == 200:
                        st.session_state.auth_token = response.json()['token']
                        st.session_state.user_info = {'username': username}
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Registration tab
    with tab2:
        with st.form("registration_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                if not new_username or not new_password:
                    st.error("Please provide both username and password")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/auth/register/",
                            json={
                                "username": new_username,
                                "email": new_email,
                                "password": new_password
                            }
                        )
                        
                        if response.status_code == 201:
                            st.session_state.auth_token = response.json()['token']
                            st.session_state.user_info = {'username': new_username}
                            st.success("Registration successful! You are now logged in.")
                            st.rerun()
                        else:
                            st.error(f"Registration failed: {response.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

def get_headers():
    auth_token = get_session()
    if auth_token:
        return {
            'Authorization': f'Token {auth_token}'
        }
    return {}

def get_column_choice(columns_info, key_prefix, multiselect=False, label="Select column"):
    """Helper function to handle column selection by name or index."""
    selection_type = st.radio(
        f"Select by",
        ["Column Name", "Column Number"],
        key=f"{key_prefix}_type"
    )
    
    if selection_type == "Column Name":
        if multiselect:
            return st.multiselect(
                label,
                options=[col['name'] for col in columns_info],
                format_func=lambda x: x,
                key=f"{key_prefix}_name"
            )
        else:
            return st.selectbox(
                label,
                options=[col['name'] for col in columns_info],
                format_func=lambda x: x,
                key=f"{key_prefix}_name"
            )
    else:
        if multiselect:
            indices = st.multiselect(
                f"{label} (by number)",
                options=list(range(len(columns_info))),
                format_func=lambda i: f"Column {i}: {columns_info[i]['name']}",
                key=f"{key_prefix}_index"
            )
            return indices
        else:
            index = st.selectbox(
                f"{label} (by number)",
                options=list(range(len(columns_info))),
                format_func=lambda i: f"Column {i}: {columns_info[i]['name']}",
                key=f"{key_prefix}_index"
            )
            return index

def main():
    # Show user info in sidebar if logged in
    if st.session_state.auth_token and st.session_state.user_info:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.user_info['username']}")
    
    # Show landing page if not logged in and not on login page
    if not get_session() and st.session_state.page == 'landing':
        landing_page()
        return
    elif not get_session():
        login_page()
        return
    
    # Main app navigation
    st.title("📊 Data Analysis Platform")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Dataset", "Dataset Overview & Cleaning", "Feature Engineering", "Visualize Data"]
    )
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.auth_token = None
        st.session_state.user_info = None
        st.session_state.page = 'landing'
        st.rerun()
    
    if page == "Upload Dataset":
        upload_dataset_page()
    elif page == "Dataset Overview & Cleaning":
        dataset_overview_page()
    elif page == "Feature Engineering":
        feature_engineering_page()
    else:
        visualize_data_page()

def upload_dataset_page():
    st.header("Upload Dataset")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"]
    )
    
    if uploaded_file is not None:
        # Get file details
        file_type = uploaded_file.type.split('/')[-1]
        if file_type == 'vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            file_type = 'xlsx'
            
        # Form for dataset details
        with st.form("dataset_form"):
            name = st.text_input("Dataset Name")
            description = st.text_area("Description")
            
            if st.form_submit_button("Upload"):
                try:
                    # Prepare the file for upload
                    files = {
                        'file': uploaded_file
                    }
                    data = {
                        'name': name,
                        'description': description,
                        'file_type': file_type
                    }
                    
                    # Upload to API with authentication
                    response = requests.post(
                        f"{API_BASE_URL}/datasets/",
                        files=files,
                        data=data,
                        headers=get_headers()
                    )
                    
                    if response.status_code == 201:
                        st.success("Dataset uploaded successfully!")
                    else:
                        st.error(f"Error uploading dataset: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def dataset_overview_page():
    st.header("Dataset Overview & Cleaning")
    
    # Add refresh button in the sidebar
    if st.sidebar.button("🔄 Refresh Data"):
        refresh_page()
    
    try:
        # Get list of datasets
        response = requests.get(
            f"{API_BASE_URL}/datasets/",
            headers=get_headers()
        )
        
        if response.status_code == 200:
            datasets = response.json()
            
            if not datasets:
                st.info("No datasets available. Please upload a dataset first.")
                return
            
            # Dataset selection
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=datasets,
                format_func=lambda x: x['name']
            )
            
            if selected_dataset:
                # Get dataset overview
                overview_response = requests.get(
                    f"{API_BASE_URL}/datasets/{selected_dataset['id']}/overview/",
                    headers=get_headers()
                )
                
                if overview_response.status_code == 200:
                    overview_data = overview_response.json()
                    
                    # Display basic information
                    st.subheader("📊 Dataset Information")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", overview_data['info']['total_rows'])
                    with col2:
                        st.metric("Total Columns", overview_data['info']['total_columns'])
                    with col3:
                        st.metric("File Type", overview_data['info']['file_type'].upper())
                    with col4:
                        total_missing = sum(overview_data['info']['missing_values'].values())
                        total_cells = overview_data['info']['total_rows'] * overview_data['info']['total_columns']
                        missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
                        st.metric("Missing Values", f"{total_missing} ({missing_percentage:.2f}%)")
                    
                    # Additional dataset information
                    st.markdown("### 📈 Dataset Statistics")
                    
                    # Create a DataFrame for all numeric statistics
                    numeric_stats_df = pd.DataFrame()
                    if overview_data.get('numeric_stats'):
                        for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                            if stat in overview_data['numeric_stats']:
                                numeric_stats_df[stat] = pd.Series(overview_data['numeric_stats'][stat])
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Numeric Statistics", "📋 Column Types", "❌ Missing Values"])
                    
                    with tab1:
                        if not numeric_stats_df.empty:
                            st.markdown("#### Numeric Columns Summary Statistics")
                            # Format the numeric values
                            formatted_df = numeric_stats_df.round(2)
                            # Add column names as a new column
                            formatted_df.insert(0, 'Column', formatted_df.index)
                            st.dataframe(formatted_df, use_container_width=True)
                        else:
                            st.info("No numeric columns found in the dataset")
                    
                    with tab2:
                        st.markdown("#### Column Data Types")
                        dtypes_df = pd.DataFrame([
                            {
                                'Column': col,
                                'Type': overview_data['info']['dtypes'].get(col, 'Unknown'),
                                'Category': ('Numeric' if col in overview_data.get('numeric_stats', {}).get('count', {})
                                           else 'Categorical' if col in overview_data.get('categorical_stats', {})
                                           else 'Datetime' if 'datetime' in overview_data['info']['dtypes'].get(col, '').lower()
                                           else 'Other')
                            }
                            for col in overview_data['info']['column_names']
                        ])
                        st.dataframe(dtypes_df, use_container_width=True)
                    
                    with tab3:
                        st.markdown("#### Missing Values Analysis")
                        missing_data = pd.DataFrame(
                            overview_data['info']['missing_values'].items(),
                            columns=['Column', 'Missing Count']
                        )
                        total_rows = overview_data['info']['total_rows']
                        missing_data['Missing Percentage'] = missing_data['Missing Count'].apply(
                            lambda x: (x / total_rows * 100) if total_rows > 0 else 0
                        ).round(2)
                        missing_data = missing_data.sort_values('Missing Percentage', ascending=False)
                        st.dataframe(missing_data, use_container_width=True)
                        
                        # Add a bar chart for missing values
                        if not missing_data.empty:
                            st.markdown("#### Missing Values Distribution")
                            missing_chart_data = missing_data[missing_data['Missing Count'] > 0]
                            if not missing_chart_data.empty:
                                st.bar_chart(data=missing_chart_data.set_index('Column')['Missing Percentage'])
                    
                    # Display sample data with tabs
                    st.markdown("### 👀 Data Preview")
                    tab1, tab2, tab3 = st.tabs(["First 5 Rows", "Last 5 Rows", "Random 5 Rows"])
                    
                    with tab1:
                        st.dataframe(pd.DataFrame(overview_data['info']['sample_data']['head']))
                    with tab2:
                        st.dataframe(pd.DataFrame(overview_data['info']['sample_data']['tail']))
                    with tab3:
                        st.dataframe(pd.DataFrame(overview_data['info']['sample_data']['random']))
                    
                    # Display column information
                    st.markdown("### 📋 Column Information")
                    
                    # Group columns by type
                    numeric_cols = []
                    categorical_cols = []
                    datetime_cols = []
                    other_cols = []
                    
                    for col in overview_data['info']['column_names']:
                        dtype = overview_data['info']['dtypes'].get(col, '')
                        # Check if column is numeric based on dtype string
                        if 'int' in dtype.lower() or 'float' in dtype.lower():
                            numeric_cols.append(col)
                        elif col in overview_data.get('categorical_stats', {}):
                            categorical_cols.append(col)
                        elif 'datetime' in dtype.lower():
                            datetime_cols.append(col)
                        else:
                            other_cols.append(col)
                    
                    # Create tabs for different column types
                    col_tabs = st.tabs([
                        f"Numeric Columns ({len(numeric_cols)})",
                        f"Categorical Columns ({len(categorical_cols)})",
                        f"Datetime Columns ({len(datetime_cols)})",
                        f"Other Columns ({len(other_cols)})"
                    ])
                    
                    # Numeric Columns Tab
                    with col_tabs[0]:
                        if numeric_cols:
                            for col in numeric_cols:
                                with st.expander(f"📊 {col} ({overview_data['info']['dtypes'].get(col, 'Unknown')})"):
                                    # Try to get statistics from numeric_stats first
                                    stats = {}
                                    if col in overview_data.get('numeric_stats', {}).get('count', {}):
                                        stats = {k: v[col] for k, v in overview_data['numeric_stats'].items()}
                                    else:
                                        # Calculate basic statistics for the column from sample data
                                        try:
                                            sample_data = pd.DataFrame(overview_data['info']['sample_data']['head'])
                                            stats = {
                                                'count': len(sample_data),
                                                'mean': sample_data[col].mean(),
                                                'std': sample_data[col].std(),
                                                'min': sample_data[col].min(),
                                                '25%': sample_data[col].quantile(0.25),
                                                '50%': sample_data[col].median(),
                                                '75%': sample_data[col].quantile(0.75),
                                                'max': sample_data[col].max()
                                            }
                                        except Exception as e:
                                            st.warning(f"Could not calculate statistics for {col}: {str(e)}")
                                            continue
                                    
                                    # Display statistics in three columns
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown("#### Basic Statistics")
                                        st.metric("Count", f"{stats.get('count', 'N/A')}")
                                        st.metric("Mean", f"{stats.get('mean', 'N/A'):.2f}" if 'mean' in stats else 'N/A')
                                        st.metric("Std Dev", f"{stats.get('std', 'N/A'):.2f}" if 'std' in stats else 'N/A')
                                    
                                    with col2:
                                        st.markdown("#### Quartile Statistics")
                                        st.metric("Minimum", f"{stats.get('min', 'N/A'):.2f}" if 'min' in stats else 'N/A')
                                        st.metric("25th Percentile", f"{stats.get('25%', 'N/A'):.2f}" if '25%' in stats else 'N/A')
                                        st.metric("Median", f"{stats.get('50%', 'N/A'):.2f}" if '50%' in stats else 'N/A')
                                        st.metric("75th Percentile", f"{stats.get('75%', 'N/A'):.2f}" if '75%' in stats else 'N/A')
                                        st.metric("Maximum", f"{stats.get('max', 'N/A'):.2f}" if 'max' in stats else 'N/A')
                                    
                                    with col3:
                                        st.markdown("#### Missing Values")
                                        missing_count = overview_data['info']['missing_values'].get(col, 0)
                                        total_rows = overview_data['info']['total_rows']
                                        missing_percentage = (missing_count / total_rows * 100) if total_rows > 0 else 0
                                        st.metric("Missing Count", missing_count)
                                        st.metric("Missing Percentage", f"{missing_percentage:.2f}%")
                                        
                                        if 'min' in stats and 'max' in stats:
                                            st.markdown("#### Range Information")
                                            range_val = stats['max'] - stats['min']
                                            st.metric("Range", f"{range_val:.2f}")
                                            if '75%' in stats and '25%' in stats:
                                                iqr = stats['75%'] - stats['25%']
                                                st.metric("IQR", f"{iqr:.2f}")
                        else:
                            st.info("No numeric columns found in the dataset.")
                    
                    # Categorical Columns Tab
                    with col_tabs[1]:
                        if categorical_cols:
                            for col in categorical_cols:
                                with st.expander(f"📝 {col}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("#### Unique Values")
                                        unique_count = overview_data['categorical_stats'][col]['unique_values']
                                        st.metric("Total Unique Values", unique_count)
                                        
                                        st.markdown("#### Top Values")
                                        top_values = overview_data['categorical_stats'][col]['top_values']
                                        for val, count in top_values.items():
                                            st.metric(str(val), count)
                                    
                                    with col2:
                                        st.markdown("#### Missing Values")
                                        missing_count = overview_data['info']['missing_values'].get(col, 0)
                                        missing_percentage = (missing_count / overview_data['info']['total_rows']) * 100
                                        st.metric("Missing Count", missing_count)
                                        st.metric("Missing Percentage", f"{missing_percentage:.2f}%")
                        else:
                            st.info("No categorical columns found in the dataset.")
                    
                    # Datetime Columns Tab
                    with col_tabs[2]:
                        if datetime_cols:
                            for col in datetime_cols:
                                with st.expander(f"📅 {col}"):
                                    st.markdown("#### Missing Values")
                                    missing_count = overview_data['info']['missing_values'].get(col, 0)
                                    missing_percentage = (missing_count / overview_data['info']['total_rows']) * 100
                                    st.metric("Missing Count", missing_count)
                                    st.metric("Missing Percentage", f"{missing_percentage:.2f}%")
                        else:
                            st.info("No datetime columns found in the dataset.")
                    
                    # Other Columns Tab
                    with col_tabs[3]:
                        if other_cols:
                            for col in other_cols:
                                with st.expander(f"❓ {col}"):
                                    st.markdown(f"Type: {overview_data['info']['dtypes'].get(col, 'Unknown')}")
                                    missing_count = overview_data['info']['missing_values'].get(col, 0)
                                    missing_percentage = (missing_count / overview_data['info']['total_rows']) * 100
                                    st.metric("Missing Count", missing_count)
                                    st.metric("Missing Percentage", f"{missing_percentage:.2f}%")
                        else:
                            st.info("No other columns found in the dataset.")
                    
                    # Data Cleaning Section
                    st.markdown("---")
                    st.subheader("🧹 Data Cleaning")
                    
                    cleaning_options = st.multiselect(
                        "Select Cleaning Operations",
                        ["Drop Columns", "Fill Missing Values", "Remove Duplicates", 
                         "Drop Missing Rows", "Rename Columns", "Change Column Type",
                         "Filter Range", "Filter Values", "Filter Condition", 
                         "Sort Values", "Sample Rows"]
                    )
                    
                    operations = []
                    
                    for option in cleaning_options:
                        st.markdown(f"### {option}")
                        
                        if option == "Drop Columns":
                            cols_to_drop = get_column_choice(
                                overview_data['info']['columns_info'],
                                'drop_cols',
                                multiselect=True,
                                label="Select columns to drop"
                            )
                            if cols_to_drop:
                                operations.append({
                                    'type': 'drop_columns',
                                    'params': {'columns': cols_to_drop}
                                })
                        
                        elif option == "Fill Missing Values":
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'fill_missing',
                                label="Select column"
                            )
                            method = st.selectbox(
                                "Fill method",
                                ["value", "mean", "median", "mode"]
                            )
                            value = None
                            if method == "value":
                                value = st.text_input("Fill value")
                            
                            operations.append({
                                'type': 'fill_missing',
                                'params': {
                                    'column': col,
                                    'method': method,
                                    'value': value
                                }
                            })
                        
                        elif option == "Remove Duplicates":
                            cols_for_dupes = get_column_choice(
                                overview_data['info']['columns_info'],
                                'duplicates',
                                multiselect=True,
                                label="Select columns to consider for duplicates (empty for all columns)"
                            )
                            operations.append({
                                'type': 'remove_duplicates',
                                'params': {'columns': cols_for_dupes}
                            })
                        
                        elif option == "Drop Missing Rows":
                            cols_for_missing = get_column_choice(
                                overview_data['info']['columns_info'],
                                'missing',
                                multiselect=True,
                                label="Select columns to consider (empty for all columns)"
                            )
                            threshold = st.slider(
                                "Minimum non-null values required",
                                1, len(overview_data['info']['columns_info']),
                                len(overview_data['info']['columns_info']) // 2
                            )
                            operations.append({
                                'type': 'drop_missing',
                                'params': {
                                    'columns': cols_for_missing,
                                    'threshold': threshold
                                }
                            })
                        
                        elif option == "Rename Columns":
                            st.write("Select columns to rename:")
                            mapping = {}
                            cols_to_rename = get_column_choice(
                                overview_data['info']['columns_info'],
                                'rename',
                                multiselect=True,
                                label="Select columns to rename"
                            )
                            for col in cols_to_rename:
                                new_name = st.text_input(
                                    f"New name for {overview_data['info']['columns_info'][col]['name'] if isinstance(col, int) else col}",
                                    key=f"rename_{col}"
                                )
                                if new_name:
                                    mapping[col] = new_name
                            if mapping:
                                operations.append({
                                    'type': 'rename_columns',
                                    'params': {'mapping': mapping}
                                })
                        
                        elif option == "Change Column Type":
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'change_type',
                                label="Select column"
                            )
                            new_type = st.selectbox(
                                "New type",
                                ["int", "float", "str", "bool", "datetime"]
                            )
                            operations.append({
                                'type': 'change_type',
                                'params': {
                                    'column': col,
                                    'new_type': new_type
                                }
                            })
                        
                        elif option == "Filter Range":
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'filter_range',
                                label="Select column"
                            )
                            
                            # Check if column is numeric
                            is_numeric = False
                            try:
                                if col in overview_data.get('numeric_stats', {}).get('count', {}):
                                    is_numeric = True
                                    stats = {k: v[col] for k, v in overview_data['numeric_stats'].items()}
                                    min_possible = float(stats['min'])
                                    max_possible = float(stats['max'])
                                else:
                                    # Try to get min and max from the data
                                    values = pd.DataFrame(overview_data['info']['sample_data']['head'])[col]
                                    if values.dtype in ['int64', 'float64']:
                                        is_numeric = True
                                        min_possible = float(values.min())
                                        max_possible = float(values.max())
                            except:
                                is_numeric = False
                            
                            if is_numeric:
                                col1, col2 = st.columns(2)
                                with col1:
                                    min_val = st.number_input("Minimum value", value=min_possible)
                                    include_min = st.checkbox("Include minimum value", value=True)
                                with col2:
                                    max_val = st.number_input("Maximum value", value=max_possible)
                                    include_max = st.checkbox("Include maximum value", value=True)
                                
                                operations.append({
                                    'type': 'filter_range',
                                    'params': {
                                        'column': col,
                                        'min_value': min_val,
                                        'max_value': max_val,
                                        'include_min': include_min,
                                        'include_max': include_max
                                    }
                                })
                            else:
                                st.warning("Selected column is not numeric. Please select a numeric column for range filtering.")
                        
                        elif option == "Filter Values":
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'filter_values',
                                label="Select column"
                            )
                            
                            # Get unique values if available
                            unique_values = []
                            if col in overview_data.get('categorical_stats', {}):
                                unique_values = list(overview_data['categorical_stats'][col]['top_values'].keys())
                            
                            values = st.multiselect(
                                "Select values to filter",
                                options=unique_values if unique_values else [],
                                help="Enter the values you want to keep or exclude"
                            )
                            
                            exclude = st.checkbox("Exclude selected values", value=False)
                            
                            if values:
                                operations.append({
                                    'type': 'filter_values',
                                    'params': {
                                        'column': col,
                                        'values': values,
                                        'exclude': exclude
                                    }
                                })
                        
                        elif option == "Filter Condition":
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'filter_condition',
                                label="Select column"
                            )
                            
                            condition = st.selectbox(
                                "Select condition",
                                ["equals", "not_equals", "greater_than", "less_than",
                                 "contains", "not_contains", "starts_with", "ends_with"]
                            )
                            
                            value = st.text_input("Enter value")
                            
                            if value:
                                # Convert value to appropriate type if possible
                                try:
                                    if overview_data['info']['dtypes'].get(col, '').startswith(('int', 'float')):
                                        value = float(value)
                                except:
                                    pass
                                
                                operations.append({
                                    'type': 'filter_condition',
                                    'params': {
                                        'column': col,
                                        'condition': condition,
                                        'value': value
                                    }
                                })
                        
                        elif option == "Sort Values":
                            cols_to_sort = get_column_choice(
                                overview_data['info']['columns_info'],
                                'sort',
                                multiselect=True,
                                label="Select columns to sort by"
                            )
                            
                            if cols_to_sort:
                                ascending = st.checkbox("Sort ascending", value=True)
                                operations.append({
                                    'type': 'sort_values',
                                    'params': {
                                        'columns': cols_to_sort,
                                        'ascending': ascending
                                    }
                                })
                        
                        elif option == "Sample Rows":
                            total_rows = overview_data['info']['total_rows']
                            n_rows = st.number_input(
                                "Number of rows to sample",
                                min_value=1,
                                max_value=total_rows,
                                value=min(1000, total_rows)
                            )
                            
                            use_random_seed = st.checkbox("Use random seed for reproducibility")
                            random_state = None
                            if use_random_seed:
                                random_state = st.number_input("Random seed", value=42)
                            
                            operations.append({
                                'type': 'sample_rows',
                                'params': {
                                    'n_rows': int(n_rows),
                                    'random_state': random_state
                                }
                            })
                    
                    if operations and st.button("Apply Cleaning Operations"):
                        # Send cleaning request
                        clean_response = requests.post(
                            f"{API_BASE_URL}/datasets/{selected_dataset['id']}/clean_data/",
                            json={'operations': operations},
                            headers=get_headers()
                        )
                        
                        if clean_response.status_code == 200:
                            st.success("Dataset cleaned successfully!")
                            # Add a refresh button after successful operation
                            if st.button("🔄 Refresh to See Changes"):
                                refresh_page()
                        else:
                            st.error(f"Error cleaning dataset: {clean_response.text}")
                
                else:
                    st.error("Error fetching dataset overview")
        
        else:
            st.error("Error fetching datasets")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def feature_engineering_page():
    st.header("Feature Engineering")
    
    # Add refresh button in the sidebar
    if st.sidebar.button("🔄 Refresh Data"):
        refresh_page()
    
    try:
        # Get list of datasets
        response = requests.get(
            f"{API_BASE_URL}/datasets/",
            headers=get_headers()
        )
        
        if response.status_code == 200:
            datasets = response.json()
            
            if not datasets:
                st.info("No datasets available. Please upload a dataset first.")
                return
            
            # Dataset selection
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=datasets,
                format_func=lambda x: x['name']
            )
            
            if selected_dataset:
                # Get dataset overview
                overview_response = requests.get(
                    f"{API_BASE_URL}/datasets/{selected_dataset['id']}/overview/",
                    headers=get_headers()
                )
                
                if overview_response.status_code == 200:
                    overview_data = overview_response.json()
                    
                    st.markdown("""
                    ### 🔧 Create New Features
                    Transform your data by creating new columns based on existing ones.
                    Choose from various operations below:
                    """)
                    
                    feature_options = st.multiselect(
                        "Select Feature Engineering Operations",
                        ["Arithmetic Operation", "Combine Text", "Apply Function", 
                         "Binning", "One-Hot Encoding", "Group Aggregation"]
                    )
                    
                    feature_operations = []
                    original_columns = overview_data['info']['column_names']
                    
                    for option in feature_options:
                        st.markdown(f"### {option}")
                        
                        if option == "Arithmetic Operation":
                            with st.expander("Create new column using arithmetic operations"):
                                new_column = st.text_input("New column name", key="arith_name")
                                cols = get_column_choice(
                                    overview_data['info']['columns_info'],
                                    'arith_cols',
                                    multiselect=True,
                                    label="Select columns for operation"
                                )
                                
                                if cols:
                                    st.write("Build your expression using:")
                                    for i, col in enumerate(cols):
                                        st.write(f"- col{i+1} for {overview_data['info']['columns_info'][col]['name'] if isinstance(col, int) else col}")
                                    
                                    expression = st.text_input(
                                        "Expression (e.g., col1 + col2 or col1 * 2)",
                                        key="arith_expr"
                                    )
                                    
                                    if new_column and expression:
                                        feature_operations.append({
                                            'type': 'arithmetic',
                                            'params': {
                                                'new_column': new_column,
                                                'expression': expression,
                                                'columns': cols
                                            }
                                        })
                        
                        elif option == "Combine Text":
                            new_column = st.text_input("New column name", key="text_name")
                            cols = get_column_choice(
                                overview_data['info']['columns_info'],
                                'text_cols',
                                multiselect=True,
                                label="Select columns to combine"
                            )
                            separator = st.text_input("Separator", " ")
                            
                            if new_column and cols:
                                feature_operations.append({
                                    'type': 'combine_text',
                                    'params': {
                                        'new_column': new_column,
                                        'columns': cols,
                                        'separator': separator
                                    }
                                })
                        
                        elif option == "Apply Function":
                            new_column = st.text_input("New column name", key="func_name")
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'func_col',
                                label="Select column"
                            )
                            
                            function = st.selectbox(
                                "Select function",
                                ["length", "lowercase", "uppercase", "log", "sqrt", 
                                 "square", "absolute", "year", "month", "day", "hour"]
                            )
                            
                            if new_column and col:
                                feature_operations.append({
                                    'type': 'apply_function',
                                    'params': {
                                        'new_column': new_column,
                                        'column': col,
                                        'function': function
                                    }
                                })
                        
                        elif option == "Binning":
                            new_column = st.text_input("New column name", key="bin_name")
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'bin_col',
                                label="Select column"
                            )
                            
                            n_bins = st.number_input("Number of bins", min_value=2, value=4)
                            
                            # Get column statistics for automatic bin ranges
                            if col in overview_data.get('numeric_stats', {}).get('count', {}):
                                stats = {k: v[col] for k, v in overview_data['numeric_stats'].items()}
                                min_val = float(stats['min'])
                                max_val = float(stats['max'])
                                
                                # Create evenly spaced bins
                                bins = list(np.linspace(min_val, max_val, n_bins + 1))
                                
                                # Create default labels
                                default_labels = [f"Bin {i+1}" for i in range(n_bins)]
                                labels_input = st.text_input(
                                    "Bin labels (comma-separated)",
                                    ",".join(default_labels)
                                )
                                labels = [label.strip() for label in labels_input.split(",")]
                                
                                if new_column and len(labels) == n_bins:
                                    feature_operations.append({
                                        'type': 'binning',
                                        'params': {
                                            'new_column': new_column,
                                            'column': col,
                                            'bins': bins,
                                            'labels': labels
                                        }
                                    })
                                else:
                                    st.warning("Number of labels must match number of bins")
                            else:
                                st.warning("Selected column must be numeric for binning")
                        
                        elif option == "One-Hot Encoding":
                            col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'onehot_col',
                                label="Select column"
                            )
                            prefix = st.text_input("Prefix for new columns", value=str(col))
                            
                            if col:
                                feature_operations.append({
                                    'type': 'one_hot_encode',
                                    'params': {
                                        'column': col,
                                        'prefix': prefix
                                    }
                                })
                        
                        elif option == "Group Aggregation":
                            new_column = st.text_input("New column name", key="agg_name")
                            group_col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'group_col',
                                label="Select grouping column"
                            )
                            value_col = get_column_choice(
                                overview_data['info']['columns_info'],
                                'value_col',
                                label="Select value column"
                            )
                            agg_function = st.selectbox(
                                "Aggregation function",
                                ["mean", "sum", "count", "min", "max"]
                            )
                            
                            if new_column and group_col and value_col:
                                feature_operations.append({
                                    'type': 'groupby_transform',
                                    'params': {
                                        'new_column': new_column,
                                        'group_column': group_col,
                                        'value_column': value_col,
                                        'agg_function': agg_function
                                    }
                                })
                    
                    if feature_operations:
                        if st.button("Create New Features"):
                            # Send feature engineering request
                            response = requests.post(
                                f"{API_BASE_URL}/datasets/{selected_dataset['id']}/feature_engineering/",
                                json={
                                    'operations': feature_operations,
                                    'original_columns': original_columns
                                },
                                headers=get_headers()
                            )
                            
                            if response.status_code == 200:
                                st.success("New features created successfully!")
                                new_columns = response.json().get('new_columns', [])
                                if new_columns:
                                    st.write("New columns created:")
                                    for col in new_columns:
                                        st.write(f"- {col}")
                                
                                # Add refresh button
                                if st.button("🔄 Refresh to See Changes"):
                                    refresh_page()
                            else:
                                st.error(f"Error creating features: {response.text}")
                
                else:
                    st.error("Error fetching dataset overview")
        
        else:
            st.error("Error fetching datasets")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def visualize_data_page():
    st.header("Visualize Data")
    
    # Add refresh button in the sidebar
    if st.sidebar.button("🔄 Refresh Data"):
        refresh_page()
    
    try:
        # Get list of datasets with authentication
        response = requests.get(
            f"{API_BASE_URL}/datasets/",
            headers=get_headers()
        )
        if response.status_code == 200:
            datasets = response.json()
            
            if not datasets:
                st.info("No datasets available. Please upload a dataset first.")
                return
            
            # Dataset selection
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=datasets,
                format_func=lambda x: x['name']
            )
            
            if selected_dataset:
                # Get dataset details
                dataset_id = selected_dataset['id']
                columns = selected_dataset['columns']
                
                # Create tabs for different visualization categories
                viz_tabs = st.tabs([
                    "📊 Basic Plots",
                    "📈 Statistical Analysis",
                    "🔍 Distribution Analysis",
                    "🔗 Correlation Analysis",
                    "📐 Regression Analysis"
                ])
                
                # Basic Plots Tab
                with viz_tabs[0]:
                    st.subheader("Basic Visualization")
                    plot_type = st.selectbox(
                        "Select Plot Type",
                        ["bar", "line", "scatter", "pie", "area", 
                         "stacked bar", "grouped bar", "bubble",
                         "radar", "polar", "waterfall"]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_column = st.selectbox(
                            "Select X-axis Column",
                            options=[col['name'] for col in columns]
                        )
                    with col2:
                        y_column = st.selectbox(
                            "Select Y-axis Column",
                            options=[col['name'] for col in columns]
                        )
                
                # Statistical Analysis Tab
                with viz_tabs[1]:
                    st.subheader("Statistical Analysis")
                    stat_plot_type = st.selectbox(
                        "Select Analysis Type",
                        ["Box Plot", "Violin Plot", "Strip Plot", 
                         "Swarm Plot", "Joint Plot", "Pair Plot",
                         "ECDF Plot", "Q-Q Plot", "Residual Plot"]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        stat_columns = st.multiselect(
                            "Select Columns for Analysis",
                            options=[col['name'] for col in columns]
                        )
                    with col2:
                        groupby_column = st.selectbox(
                            "Group By (Optional)",
                            options=["None"] + [col['name'] for col in columns]
                        )
                
                # Distribution Analysis Tab
                with viz_tabs[2]:
                    st.subheader("Distribution Analysis")
                    dist_plot_type = st.selectbox(
                        "Select Distribution Plot",
                        ["Histogram", "KDE Plot", "Dist Plot", 
                         "Rug Plot", "2D Histogram", "2D KDE",
                         "Empirical CDF", "Normal Probability Plot"]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        dist_column = st.selectbox(
                            "Select Column for Distribution",
                            options=[col['name'] for col in columns]
                        )
                    with col2:
                        bins = st.slider("Number of Bins", 5, 100, 30)
                        kde = st.checkbox("Show KDE", value=True)
                
                # Correlation Analysis Tab
                with viz_tabs[3]:
                    st.subheader("Correlation Analysis")
                    corr_plot_type = st.selectbox(
                        "Select Correlation Plot",
                        ["Correlation Matrix", "Scatter Matrix", 
                         "Pairwise Correlation", "Correlation Heatmap",
                         "Cross Correlation", "Autocorrelation"]
                    )
                    
                    corr_columns = st.multiselect(
                        "Select Columns for Correlation",
                        options=[col['name'] for col in columns]
                    )
                    
                    corr_method = st.selectbox(
                        "Correlation Method",
                        ["pearson", "spearman", "kendall"]
                    )
                
                # Regression Analysis Tab
                with viz_tabs[4]:
                    st.subheader("Regression Analysis")
                    reg_plot_type = st.selectbox(
                        "Select Regression Type",
                        ["Linear Regression", "Polynomial Regression",
                         "Lowess Regression", "Robust Regression",
                         "Logistic Regression", "Ridge Plot"]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_reg = st.selectbox(
                            "Select Independent Variable (X)",
                            options=[col['name'] for col in columns]
                        )
                    with col2:
                        y_reg = st.selectbox(
                            "Select Dependent Variable (Y)",
                            options=[col['name'] for col in columns]
                        )
                    
                    # Additional regression options
                    reg_options = {}
                    if reg_plot_type == "Polynomial Regression":
                        reg_options['degree'] = st.slider("Polynomial Degree", 1, 5, 2)
                    elif reg_plot_type == "Lowess Regression":
                        reg_options['frac'] = st.slider("Smoothing Factor", 0.1, 1.0, 0.6)
                    
                    # Common options for all regression plots
                    reg_options['ci'] = st.slider("Confidence Interval", 0, 100, 95)
                    reg_options['scatter'] = st.checkbox("Show Scatter Points", value=True)
                
                # Generate Plot Button
                if st.button("Generate Plot"):
                    # Prepare plot data based on selected tab and options
                    plot_data = {
                        'plot_type': plot_type,
                        'x_column': x_column,
                        'y_column': y_column,
                        'options': {
                            'stat_plot_type': stat_plot_type if 'stat_plot_type' in locals() else None,
                            'stat_columns': stat_columns if 'stat_columns' in locals() else None,
                            'groupby_column': groupby_column if 'groupby_column' in locals() else None,
                            'dist_plot_type': dist_plot_type if 'dist_plot_type' in locals() else None,
                            'dist_column': dist_column if 'dist_column' in locals() else None,
                            'bins': bins if 'bins' in locals() else None,
                            'kde': kde if 'kde' in locals() else None,
                            'corr_plot_type': corr_plot_type if 'corr_plot_type' in locals() else None,
                            'corr_columns': corr_columns if 'corr_columns' in locals() else None,
                            'corr_method': corr_method if 'corr_method' in locals() else None,
                            'reg_plot_type': reg_plot_type if 'reg_plot_type' in locals() else None,
                            'reg_options': reg_options if 'reg_options' in locals() else None
                        }
                    }
                    
                    # Send plot request to backend
                    response = requests.post(
                        f"{API_BASE_URL}/datasets/{dataset_id}/generate_plot/",
                        json=plot_data,
                        headers=get_headers()
                    )
                    
                    if response.status_code == 200:
                        plot_response = response.json()
                        # Display the plot
                        st.image(
                            plot_response['plot'],
                            use_column_width=True
                        )
                        
                        # Display additional statistics if available
                        if 'statistics' in plot_response:
                            st.subheader("Statistical Summary")
                            st.json(plot_response['statistics'])
                    else:
                        st.error(f"Error generating plot: {response.text}")
        
        else:
            st.error("Error fetching datasets")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
