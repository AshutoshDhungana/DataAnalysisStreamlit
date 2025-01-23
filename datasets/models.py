from django.db import models
from django.contrib.auth.models import User
import os

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    file_type = models.CharField(max_length=50)  # e.g., 'csv', 'excel'
    columns = models.JSONField(default=list)  # Store column names and types
    row_count = models.IntegerField(default=0)
    version = models.IntegerField(default=1)  # Track dataset versions
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} ({self.file_type})"
    
    def save(self, *args, **kwargs):
        if self.pk:  # If this is an update
            self.version += 1
        super().save(*args, **kwargs)
    
    def get_new_filename(self, prefix=''):
        """Generate a new filename for the dataset."""
        base_name = os.path.splitext(os.path.basename(self.file.name))[0]
        extension = os.path.splitext(self.file.name)[1]
        return f"datasets/{prefix}_{base_name}_v{self.version}{extension}"
