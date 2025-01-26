from rest_framework import serializers
from .models import Dataset

class DatasetSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True)
    file_type = serializers.CharField(write_only=True)

    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'file', 'uploaded_at', 'updated_at', 
                 'user', 'file_type', 'columns', 'row_count']
        read_only_fields = ['uploaded_at', 'updated_at', 'user', 'columns', 'row_count']

    def create(self, validated_data):
        # Remove file and file_type as they're not model fields
        file = validated_data.pop('file')
        file_type = validated_data.pop('file_type')
        
        # Create the dataset instance
        instance = Dataset.objects.create(**validated_data)
        
        # Store file and file_type in context for the view to handle
        self.context['file'] = file
        self.context['file_type'] = file_type
        
        return instance 