from rest_framework import serializers
from .models import Dataset

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'file', 'uploaded_at', 'updated_at', 
                 'user', 'file_type', 'columns', 'row_count']
        read_only_fields = ['uploaded_at', 'updated_at', 'user', 'columns', 'row_count'] 