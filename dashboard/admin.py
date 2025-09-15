import requests, json
from urllib.request import urlopen
from django import forms
from django.utils.html import format_html
from django.contrib import admin
from .models import Project, AggregateMetric, ProjectMetric, ProjectMetricData
from . import utils
from django.utils import timezone

API_URL = "https://api.carboncopy.news/projects"


class ProjectSelectForm(forms.ModelForm):
    project_selector = forms.ChoiceField(label="Select Project", required=True)
    _cached_projects = []  # class-level cache

    class Meta:
        model = Project
        fields = []  # hide actual model fields

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load and cache projects if not already cached
        if not ProjectSelectForm._cached_projects:
            try:
                response = requests.get(API_URL, timeout=5)
                response.raise_for_status()
                ProjectSelectForm._cached_projects = response.json().get("projects", [])
            except Exception as e:
                ProjectSelectForm._cached_projects = []
                self.fields["project_selector"].help_text = f"Error loading projects: {e}"

        # Populate choices from cached projects
        choices = [(p["id"], p["name"]) for p in ProjectSelectForm._cached_projects]
        self.fields["project_selector"].choices = choices

    def save(self, commit=True):
        selected_id = self.cleaned_data["project_selector"]

        # Look up selected project from cached data
        selected_project = next(
            (p for p in ProjectSelectForm._cached_projects if str(p["id"]) == str(selected_id)),
            None
        )

        if selected_project:
            self.instance.baserow_id = selected_project["id"]
            self.instance.name = selected_project["name"]
            self.instance.logo_url = selected_project.get("logo", "")

        return super().save(commit=commit)

class ProjectMetricDataForm(forms.ModelForm):
    project_metrics = forms.ModelMultipleChoiceField(
        queryset=ProjectMetric.objects.all(),
        required=True,
        label="Metrics"
    )
    value = forms.FloatField(required=True, label="Value")
    date = forms.DateField(
        required=True,
        widget=forms.SelectDateWidget(years=range(2000, timezone.now().year + 1)),
        label="Date",
    )

    class Meta:
        model = ProjectMetricData
        fields = ["project_metrics", "value", "date"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom label: "Metric name (Project name)"
        self.fields["project_metrics"].label_from_instance = (
            lambda obj: f"{obj.name} ({', '.join(p.name for p in obj.projects.all())})"
        )

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    form = ProjectSelectForm
    list_display = ("name", "baserow_id", "logo_url")

@admin.register(AggregateMetric)
class AggregateMetricAdmin(admin.ModelAdmin):
    list_display = ('name', 'unit', 'format', 'type', 'description', 'chart')
    search_fields = ('name', 'unit', 'description')
    list_filter = ('type', 'format')

@admin.register(ProjectMetric)
class ProjectMetricAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'get_projects', 'unit', 'category', 
        'current_value', 'current_value_date'
    )
    list_filter = ('category', 'format', 'projects', 'aggregate_metric')
    search_fields = ('name', 'unit', 'db_id', 'description')

    readonly_fields = ('current_value', 'current_value_date')

    actions = ['update_impact_data']

    def get_projects(self, obj):
        return ", ".join(p.name for p in obj.projects.all())
    
    get_projects.short_description = "Projects"

    def update_impact_data(self, request, queryset):
        from . import update_metrics

        for metric in queryset:
            for project in metric.projects.all():
                baserow_id = project.baserow_id
                db_id = metric.db_id

            try:
                # 1) Call the function in utils.py
                result = utils.get_baserow_impact_data(baserow_id)

                # 2) Extract JSON file URL
                json_url = result["Impact Metrics JSON"][0]["url"]

                # 3) Download JSON and extract "source"
                response = urlopen(json_url)
                data = json.loads(response.read())
                source_value = data['impact_data'][0]['source']

                if not source_value:
                    self.message_user(request, f"No source found for {metric.name}", level="error")
                    continue

                # 4) Call refresh function based on source_value
                func_name = f"refresh_{source_value}"
                if not hasattr(update_metrics, func_name):
                    self.message_user(request, f"Function {func_name} not found in update_metrics.py", level="error")
                    continue
                func = getattr(update_metrics, func_name)
                value = func(data['impact_data'][0], db_id)  # Pass full impact JSON

                if not isinstance(value, float):
                    self.message_user(request, f"Invalid return value from {func_name}", level="error")
                    continue
               
                now = timezone.now()

                # Step 5: Get last record for this metric
                last_record = (
                    ProjectMetricData.objects
                    .filter(project_metrics=metric)
                    .order_by("-date")
                    .first()
                )

                if last_record is None:
                    # No prior data → store the actual value
                    record_value = value
                else:
                    # Store the delta (can be zero or negative)
                    record_value = value - metric.current_value

                # Step 6: Update ProjectMetricData table              
                record = ProjectMetricData.objects.create(
                    value=round(record_value, 2),
                    date=now,
                )
                record.project_metrics.add(metric)

                # Step 7: Update ProjectMetric table
                metric.current_value = round(value, 2)
                metric.current_value_date = now
                metric.save(update_fields=["current_value", "current_value_date"])

                self.message_user(
                    request,
                    f"Updated {metric.name}: stored {record_value} (delta mode), latest value = {value}"
                )

            except Exception as e:
                self.message_user(
                    request,
                    f"Error updating '{metric.name}': {e}",
                    level='error'
                )

    update_impact_data.short_description = "Update impact data"

@admin.register(ProjectMetricData)
class ProjectMetricDataAdmin(admin.ModelAdmin):
    form = ProjectMetricDataForm
    list_display = ("get_metrics", "value", "date")

    def get_metrics(self, obj):
        return ", ".join(
            f"{pm.name} ({', '.join(p.name for p in pm.projects.all())})"
            for pm in obj.project_metrics.all()
        )
    
    get_metrics.short_description = "Metrics"

    def has_view_permission(self, request, obj=None):
        # Prevent access to the changelist page
        return False

    def changelist_view(self, request, extra_context=None):
        # Redirect the list view to the add form
        from django.shortcuts import redirect
        return redirect("admin:dashboard_projectmetricdata_add")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)

    def save_related(self, request, form, formsets, change):
        super().save_related(request, form, formsets, change)

        obj = form.instance  # The saved ProjectMetricData
        metrics = obj.project_metrics.all()

        for metric in metrics:
            if metric.current_value is None:
                delta_value = obj.value
            else:
                delta_value = obj.value - metric.current_value

            metric.current_value = round(
                metric.current_value + delta_value if metric.current_value else obj.value, 2
            )
            metric.current_value_date = obj.date
            metric.save(update_fields=["current_value", "current_value_date"])

            self.message_user(
                request,
                f"Added {delta_value} (delta) to {metric.name}, latest value = {metric.current_value}"
            )