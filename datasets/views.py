import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO
import base64
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.core.files.base import ContentFile
from .models import Dataset
from .serializers import DatasetSerializer
import os
import uuid
from datetime import datetime
import seaborn as sns
from scipy import stats

class DatasetViewSet(viewsets.ModelViewSet):
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Dataset.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        dataset = serializer.save(user=self.request.user)
        self._process_dataset(dataset)
    
    def _process_dataset(self, dataset):
        """Process uploaded dataset to extract metadata."""
        try:
            if dataset.file_type.lower() == 'csv':
                df = pd.read_csv(dataset.file.path)
            elif dataset.file_type.lower() in ['xlsx', 'excel']:
                df = pd.read_excel(dataset.file.path)
            else:
                raise ValueError(f"Unsupported file type: {dataset.file_type}")
            
            # Update dataset metadata
            dataset.columns = [
                {'name': col, 'type': str(df[col].dtype)}
                for col in df.columns
            ]
            dataset.row_count = len(df)
            dataset.save()
            
        except Exception as e:
            dataset.delete()
            raise ValueError(f"Error processing dataset: {str(e)}")
    
    @action(detail=True, methods=['get'])
    def overview(self, request, pk=None):
        """Get dataset overview including statistics and sample data."""
        dataset = self.get_object()
        try:
            # Read the dataset
            if dataset.file_type.lower() == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            # Get column information with indices
            columns_info = []
            for i, col in enumerate(df.columns):
                col_name = str(col)
                if "Unnamed" in col_name:
                    col_name = f"Column_{i} (Unnamed)"
                columns_info.append({
                    'index': i,
                    'name': col_name,
                    'original_name': str(col)
                })
            
            # Get basic information
            info = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns_info': columns_info,
                'column_names': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': {
                    'head': df.head(5).to_dict(orient='records'),
                    'tail': df.tail(5).to_dict(orient='records'),
                    'random': df.sample(min(5, len(df))).to_dict(orient='records')
                }
            }
            
            # Get numerical column statistics
            numeric_stats = {}
            try:
                # Get numeric columns based on dtype
                numeric_cols = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
                if len(numeric_cols) > 0:
                    numeric_stats = df[numeric_cols].describe().to_dict()
            except Exception as e:
                print(f"Error in numeric stats: {str(e)}")
                pass
            
            # Get categorical column information
            categorical_stats = {}
            try:
                # Get categorical columns (exclude numeric and datetime)
                categorical_cols = df.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'datetime64']).columns
                categorical_stats = {
                    col: {
                        'unique_values': df[col].nunique(),
                        'top_values': df[col].value_counts().head(5).to_dict()
                    }
                    for col in categorical_cols
                }
            except Exception as e:
                print(f"Error in categorical stats: {str(e)}")
                pass
            
            # Update info to include column categories
            info['column_categories'] = {
                'numeric': [str(col) for col in numeric_cols],
                'categorical': [str(col) for col in categorical_cols],
                'datetime': [str(col) for col in df.select_dtypes(include=['datetime64']).columns]
            }
            
            return Response({
                'info': info,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['post'])
    def clean_data(self, request, pk=None):
        """Clean dataset based on specified operations."""
        dataset = self.get_object()
        operations = request.data.get('operations', [])
        
        try:
            # Read the dataset
            if dataset.file_type.lower() == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            # Apply cleaning operations
            for operation in operations:
                op_type = operation.get('type')
                params = operation.get('params', {})
                
                # Helper function to get column name from index or name
                def get_column(col_identifier):
                    if isinstance(col_identifier, int):
                        return df.columns[col_identifier]
                    return col_identifier
                
                if op_type == 'drop_columns':
                    columns = [get_column(col) for col in params.get('columns', [])]
                    df = df.drop(columns=columns)
                
                elif op_type == 'fill_missing':
                    column = get_column(params.get('column'))
                    method = params.get('method')
                    value = params.get('value')
                    
                    if method == 'value':
                        df[column] = df[column].fillna(value)
                    elif method == 'mean':
                        df[column] = df[column].fillna(df[column].mean())
                    elif method == 'median':
                        df[column] = df[column].fillna(df[column].median())
                    elif method == 'mode':
                        df[column] = df[column].fillna(df[column].mode()[0])
                
                elif op_type == 'drop_missing':
                    columns = [get_column(col) for col in params.get('columns', [])]
                    threshold = params.get('threshold')
                    if columns:
                        df = df.dropna(subset=columns, thresh=threshold)
                    else:
                        df = df.dropna(thresh=threshold)
                
                elif op_type == 'remove_duplicates':
                    columns = [get_column(col) for col in params.get('columns', [])]
                    if columns:
                        df = df.drop_duplicates(subset=columns)
                    else:
                        df = df.drop_duplicates()
                
                elif op_type == 'rename_columns':
                    mapping = {get_column(old): new for old, new in params.get('mapping', {}).items()}
                    df = df.rename(columns=mapping)
                
                elif op_type == 'change_type':
                    column = get_column(params.get('column'))
                    new_type = params.get('new_type')
                    df[column] = df[column].astype(new_type)
                
                elif op_type == 'filter_range':
                    column = get_column(params.get('column'))
                    min_val = params.get('min_value')
                    max_val = params.get('max_value')
                    include_min = params.get('include_min', True)
                    include_max = params.get('include_max', True)
                    
                    if min_val is not None and max_val is not None:
                        if include_min and include_max:
                            mask = (df[column] >= min_val) & (df[column] <= max_val)
                        elif include_min:
                            mask = (df[column] >= min_val) & (df[column] < max_val)
                        elif include_max:
                            mask = (df[column] > min_val) & (df[column] <= max_val)
                        else:
                            mask = (df[column] > min_val) & (df[column] < max_val)
                        df = df[mask]
                    elif min_val is not None:
                        df = df[df[column] >= min_val if include_min else df[column] > min_val]
                    elif max_val is not None:
                        df = df[df[column] <= max_val if include_max else df[column] < max_val]
                
                elif op_type == 'filter_values':
                    column = get_column(params.get('column'))
                    values = params.get('values', [])
                    exclude = params.get('exclude', False)
                    
                    if exclude:
                        df = df[~df[column].isin(values)]
                    else:
                        df = df[df[column].isin(values)]
                
                elif op_type == 'filter_condition':
                    column = get_column(params.get('column'))
                    condition = params.get('condition')
                    value = params.get('value')
                    
                    if condition == 'equals':
                        df = df[df[column] == value]
                    elif condition == 'not_equals':
                        df = df[df[column] != value]
                    elif condition == 'greater_than':
                        df = df[df[column] > value]
                    elif condition == 'less_than':
                        df = df[df[column] < value]
                    elif condition == 'contains':
                        df = df[df[column].astype(str).str.contains(str(value), na=False)]
                    elif condition == 'not_contains':
                        df = df[~df[column].astype(str).str.contains(str(value), na=False)]
                    elif condition == 'starts_with':
                        df = df[df[column].astype(str).str.startswith(str(value))]
                    elif condition == 'ends_with':
                        df = df[df[column].astype(str).str.endswith(str(value))]
                
                elif op_type == 'sort_values':
                    columns = params.get('columns', [])
                    ascending = params.get('ascending', True)
                    df = df.sort_values(by=columns, ascending=ascending)
                
                elif op_type == 'sample_rows':
                    n_rows = params.get('n_rows')
                    random_state = params.get('random_state')
                    if n_rows:
                        df = df.sample(n=min(n_rows, len(df)), random_state=random_state)
            
            # Save the cleaned dataset
            buffer = StringIO()
            if dataset.file_type.lower() == 'csv':
                df.to_csv(buffer, index=False)
            else:
                df.to_excel(buffer, index=False)
            
            # Update the file
            buffer.seek(0)
            content = ContentFile(buffer.getvalue().encode('utf-8'))
            
            # Generate new filename using the model's method
            new_filename = dataset.get_new_filename(prefix='cleaned')
            dataset.file.save(new_filename, content, save=True)
            
            # Update metadata
            dataset.columns = [
                {'name': col, 'type': str(df[col].dtype)}
                for col in df.columns
            ]
            dataset.row_count = len(df)
            dataset.save()
            
            return Response({
                'message': 'Dataset cleaned successfully',
                'rows': len(df),
                'columns': list(df.columns)
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['post'])
    def feature_engineering(self, request, pk=None):
        """Create new features based on existing columns."""
        dataset = self.get_object()
        operations = request.data.get('operations', [])
        original_columns = request.data.get('original_columns', [])
        
        try:
            df = pd.read_csv(dataset.file.path)
            new_columns = []
            
            for operation in operations:
                op_type = operation['type']
                params = operation['params']
                
                if op_type == 'arithmetic':
                    # Replace column references with actual column names
                    expr = params['expression']
                    for i, col in enumerate(params['columns']):
                        col_name = df.columns[col] if isinstance(col, int) else col
                        expr = expr.replace(f'col{i+1}', f"df['{col_name}']")
                    
                    # Safely evaluate the expression
                    try:
                        df[params['new_column']] = eval(expr)
                        new_columns.append(params['new_column'])
                    except Exception as e:
                        return Response({
                            'error': f"Error in arithmetic operation: {str(e)}"
                        }, status=status.HTTP_400_BAD_REQUEST)
                
                elif op_type == 'apply_function':
                    col = df.columns[params['column']] if isinstance(params['column'], int) else params['column']
                    
                    if params['function'] == 'custom':
                        try:
                            # Create a safe namespace for custom function execution
                            safe_globals = {
                                'pd': pd,
                                'np': np,
                                'df': df[col],
                                'result': None
                            }
                            
                            # Execute the custom code in the safe namespace
                            exec(params['custom_code'], safe_globals)
                            
                            if safe_globals.get('result') is not None:
                                df[params['new_column']] = safe_globals['result']
                                new_columns.append(params['new_column'])
                            else:
                                return Response({
                                    'error': "Custom function did not set the 'result' variable"
                                }, status=status.HTTP_400_BAD_REQUEST)
                        
                        except Exception as e:
                            return Response({
                                'error': f"Error in custom function: {str(e)}"
                            }, status=status.HTTP_400_BAD_REQUEST)
                    
                    else:
                        # Handle built-in functions
                        try:
                            if params['function'] == 'length':
                                df[params['new_column']] = df[col].astype(str).str.len()
                            elif params['function'] in ['lowercase', 'uppercase']:
                                func = str.lower if params['function'] == 'lowercase' else str.upper
                                df[params['new_column']] = df[col].astype(str).map(func)
                            elif params['function'] == 'log':
                                df[params['new_column']] = np.log(df[col])
                            elif params['function'] == 'sqrt':
                                df[params['new_column']] = np.sqrt(df[col])
                            elif params['function'] == 'square':
                                df[params['new_column']] = np.square(df[col])
                            elif params['function'] == 'absolute':
                                df[params['new_column']] = np.abs(df[col])
                            elif params['function'] in ['year', 'month', 'day', 'hour']:
                                df[params['new_column']] = pd.to_datetime(df[col])\
                                    .dt.__getattribute__(params['function'])
                            
                            new_columns.append(params['new_column'])
                        
                        except Exception as e:
                            return Response({
                                'error': f"Error applying {params['function']}: {str(e)}"
                            }, status=status.HTTP_400_BAD_REQUEST)
                
                elif op_type == 'combine_text':
                    new_column = params.get('new_column')
                    columns = [col for col in params.get('columns', []) if isinstance(col, str)]
                    separator = params.get('separator', ' ')
                    df[new_column] = df[columns].astype(str).agg(separator.join, axis=1)
                    new_columns.append(new_column)
                
                elif op_type == 'binning':
                    new_column = params.get('new_column')
                    column = col
                    bins = params.get('bins')
                    labels = params.get('labels')
                    
                    df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
                    new_columns.append(new_column)
                
                elif op_type == 'one_hot_encode':
                    column = col
                    prefix = params.get('prefix', column)
                    
                    # Create dummy variables
                    dummies = pd.get_dummies(df[column], prefix=prefix)
                    df = pd.concat([df, dummies], axis=1)
                    new_columns.extend(dummies.columns.tolist())
                
                elif op_type == 'groupby_transform':
                    new_column = params.get('new_column')
                    group_column = col
                    value_column = params.get('value_column')
                    agg_function = params.get('agg_function')
                    
                    if agg_function == 'mean':
                        df[new_column] = df.groupby(group_column)[value_column].transform('mean')
                    elif agg_function == 'sum':
                        df[new_column] = df.groupby(group_column)[value_column].transform('sum')
                    elif agg_function == 'count':
                        df[new_column] = df.groupby(group_column)[value_column].transform('count')
                    elif agg_function == 'min':
                        df[new_column] = df.groupby(group_column)[value_column].transform('min')
                    elif agg_function == 'max':
                        df[new_column] = df.groupby(group_column)[value_column].transform('max')
                    new_columns.append(new_column)
            
            # Save the updated dataset with a new filename
            new_filename = dataset.get_new_filename(prefix='featured')
            
            # Save to CSV
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            content = ContentFile(buffer.getvalue().encode('utf-8'))
            dataset.file.save(new_filename, content, save=True)
            
            # Update metadata
            dataset.columns = [
                {'name': col, 'type': str(df[col].dtype)}
                for col in df.columns
            ]
            dataset.row_count = len(df)
            dataset.save()
            
            return Response({
                'message': 'Features created successfully',
                'new_columns': new_columns
            })
        
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def generate_plot(self, request, pk=None):
        """Generate plot based on user parameters."""
        dataset = self.get_object()
        plot_type = request.data.get('plot_type', '').lower()
        x_column = request.data.get('x_column')
        y_column = request.data.get('y_column')
        options = request.data.get('options', {})
        
        try:
            # Read the dataset
            df = pd.read_csv(dataset.file.path) if dataset.file_type.lower() == 'csv' else pd.read_excel(dataset.file.path)
            
            # Create figure with appropriate size
            plt.figure(figsize=(12, 8))
            
            # Initialize statistics dictionary
            statistics = {}
            
            # Basic Plots
            if plot_type in ["bar", "line", "scatter", "pie", "area"]:
                if plot_type == "bar":
                    plt.bar(df[x_column], df[y_column])
                elif plot_type == "line":
                    plt.plot(df[x_column], df[y_column])
                elif plot_type == "scatter":
                    plt.scatter(df[x_column], df[y_column])
                elif plot_type == "area":
                    plt.fill_between(df[x_column], df[y_column])
                elif plot_type == "pie":
                    plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%')
            
            # Statistical Analysis
            elif options.get('stat_plot_type'):
                stat_type = options['stat_plot_type']
                columns = options.get('stat_columns', [])
                group_col = options.get('groupby_column')
                
                if stat_type == "Box Plot":
                    sns.boxplot(data=df, x=group_col if group_col != "None" else None, 
                              y=columns[0] if len(columns) > 0 else None)
                elif stat_type == "Violin Plot":
                    sns.violinplot(data=df, x=group_col if group_col != "None" else None,
                                 y=columns[0] if len(columns) > 0 else None)
                elif stat_type == "Strip Plot":
                    sns.stripplot(data=df, x=group_col if group_col != "None" else None,
                                y=columns[0] if len(columns) > 0 else None)
                elif stat_type == "Swarm Plot":
                    sns.swarmplot(data=df, x=group_col if group_col != "None" else None,
                                y=columns[0] if len(columns) > 0 else None)
                elif stat_type == "Joint Plot":
                    if len(columns) >= 2:
                        g = sns.jointplot(data=df, x=columns[0], y=columns[1])
                        plt.close()  # Close the extra figure created by jointplot
                        return Response({'plot': self._fig_to_base64(g.figure)})
                elif stat_type == "Pair Plot":
                    if len(columns) >= 2:
                        g = sns.pairplot(df[columns])
                        plt.close()  # Close the extra figure created by pairplot
                        return Response({'plot': self._fig_to_base64(g.figure)})
                
                # Calculate basic statistics
                statistics['basic_stats'] = df[columns].describe().to_dict() if columns else {}
            
            # Distribution Analysis
            elif options.get('dist_plot_type'):
                dist_type = options['dist_plot_type']
                column = options.get('dist_column')
                bins = options.get('bins', 30)
                kde = options.get('kde', True)
                
                if dist_type == "Histogram":
                    sns.histplot(data=df, x=column, bins=bins, kde=kde)
                elif dist_type == "KDE Plot":
                    sns.kdeplot(data=df, x=column)
                elif dist_type == "Dist Plot":
                    sns.distplot(df[column], bins=bins, kde=kde)
                elif dist_type == "2D Histogram":
                    if x_column and y_column:
                        plt.hist2d(df[x_column], df[y_column], bins=bins)
                        plt.colorbar()
                
                # Calculate distribution statistics
                statistics['distribution_stats'] = {
                    'mean': float(df[column].mean()),
                    'median': float(df[column].median()),
                    'std': float(df[column].std()),
                    'skew': float(df[column].skew()),
                    'kurtosis': float(df[column].kurtosis())
                }
            
            # Correlation Analysis
            elif options.get('corr_plot_type'):
                corr_type = options['corr_plot_type']
                columns = options.get('corr_columns', [])
                method = options.get('corr_method', 'pearson')
                
                if corr_type == "Correlation Matrix":
                    corr_matrix = df[columns].corr(method=method)
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                elif corr_type == "Scatter Matrix":
                    pd.plotting.scatter_matrix(df[columns], diagonal='kde')
                elif corr_type == "Pairwise Correlation":
                    g = sns.pairplot(df[columns], diag_kind='kde')
                    plt.close()
                    return Response({'plot': self._fig_to_base64(g.figure)})
                
                # Calculate correlation statistics
                statistics['correlation_matrix'] = df[columns].corr(method=method).to_dict()
            
            # Regression Analysis
            elif options.get('reg_plot_type'):
                reg_type = options['reg_plot_type']
                x_reg = options.get('x_reg')
                y_reg = options.get('y_reg')
                reg_options = options.get('reg_options', {})
                
                if reg_type == "Linear Regression":
                    sns.regplot(data=df, x=x_reg, y=y_reg, 
                              ci=reg_options.get('ci', 95),
                              scatter=reg_options.get('scatter', True))
                elif reg_type == "Polynomial Regression":
                    sns.regplot(data=df, x=x_reg, y=y_reg,
                              order=reg_options.get('degree', 2),
                              ci=reg_options.get('ci', 95),
                              scatter=reg_options.get('scatter', True))
                elif reg_type == "Lowess Regression":
                    sns.regplot(data=df, x=x_reg, y=y_reg,
                              lowess=True, ci=None,
                              scatter=reg_options.get('scatter', True))
                
                # Calculate regression statistics
                slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_reg], df[y_reg])
                statistics['regression_stats'] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'std_err': float(std_err)
                }
            
            # Set labels and title
            if plot_type not in ['pie']:
                if x_column:
                    plt.xlabel(x_column)
                if y_column:
                    plt.ylabel(y_column)
            
            # Set title based on plot type
            title = options.get('stat_plot_type') or options.get('dist_plot_type') or \
                   options.get('corr_plot_type') or options.get('reg_plot_type') or \
                   plot_type.title()
            plt.title(f'{title} Plot')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to base64
            plot_base64 = self._fig_to_base64(plt.gcf())
            plt.close()
            
            return Response({
                'plot': plot_base64,
                'statistics': statistics
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/png;base64,{image_base64}' 