from django.db import models
from django.contrib.auth.models import User
import pandas as pd
from django.contrib.postgres.fields import ArrayField

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    version = models.IntegerField(default=1)
    file_type = models.CharField(max_length=10, default='csv')  # Store the original file format
    
    # Store dataset metadata
    columns = models.JSONField(default=list)  # Column names and types
    row_count = models.IntegerField(default=0)
    
    # Store the actual data
    data = models.JSONField(default=dict)  # Store the dataset as JSON
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} (v{self.version})"
    
    def save(self, *args, **kwargs):
        if self.pk:  # If this is an update
            self.version += 1
        super().save(*args, **kwargs)
    
    def set_data_from_df(self, df, file_type=None):
        """Convert DataFrame to JSON and store in database."""
        # Replace NaN values with None (null in JSON) before converting to dict
        df = df.replace({float('nan'): None})
        
        # Store file type with default fallback
        self.file_type = (file_type or 'csv').lower()
        
        # Convert DataFrame to JSON format
        self.data = {
            'data': df.to_dict(orient='records'),
        }
        
        # Store column information
        self.columns = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]
        
        # Store row count
        self.row_count = len(df)
    
    def get_data_as_df(self):
        """Convert stored JSON data back to DataFrame."""
        if not self.data:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_records(self.data['data'])
        
        # Restore original data types using column information
        for col_info in self.columns:
            col = col_info['name']
            dtype = col_info['type']
            try:
                if 'datetime' in dtype.lower():
                    df[col] = pd.to_datetime(df[col])
                elif 'int' in dtype.lower():
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif 'float' in dtype.lower():
                    df[col] = pd.to_numeric(df[col], downcast='float')
            except:
                continue
                
        return df
