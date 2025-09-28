import requests, json, csv
from urllib.request import urlopen
from django import forms
from django.utils.html import format_html
from django.contrib import admin, messages
from .models import Project, AggregateMetric, ProjectMetric, ProjectMetricData, APIKey
from . import utils
from django.utils import timezone
from django.shortcuts import render, redirect

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
            self.instance.slug = selected_project["slug"]

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
    list_display = ("name", "slug", "baserow_id", "logo_url")

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

                # 3) Download JSON
                response = urlopen(json_url)
                data = json.loads(response.read())

                if "impact_data" not in data or not isinstance(data["impact_data"], list):
                    self.message_user(request, f"No impact_data found for {metric.name}", level="error")
                    continue

                # --- find the correct source block where db_id matches ---
                matched_item = None
                matched_metric = None
                for item in data["impact_data"]:
                    for m in item.get("metrics", []):
                        if m.get("db_id") == db_id:
                            matched_item = item
                            matched_metric = m.get("db_id")
                            break
                    if matched_item:
                        break

                if not matched_item or not matched_metric:
                    self.message_user(request, f"No impact_data entry found for db_id={db_id} ({metric.name})", level="error")
                    continue

                source_value = matched_item.get("source")

                if not source_value:
                    self.message_user(request, f"No source found for {metric.name}", level="error")
                    continue

                # 4) Resolve function dynamically
                func_name = f"refresh_{source_value}"
                if not hasattr(update_metrics, func_name):
                    self.message_user(request, f"Function {func_name} not found in update_metrics.py", level="error")
                    continue
                func = getattr(update_metrics, func_name)

                # pass both the matched source block and the specific metric config
                value = func(matched_item, matched_metric)
                print(type(value))
                if not isinstance(value, (float, int)):
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
                    record_value = value
                else:
                    record_value = value - metric.current_value

                # Step 6: Store record
                record = ProjectMetricData.objects.create(
                    value=round(record_value, 2),
                    date=now,
                )
                record.project_metrics.add(metric)

                # Step 7: Update ProjectMetric
                metric.current_value = round(value, 2)
                metric.current_value_date = now
                metric.save(update_fields=["current_value", "current_value_date"])

                self.message_user(
                    request,
                    f"Updated {metric.name}: stored delta {record_value}, latest value = {value}"
                )

            except Exception as e:
                self.message_user(
                    request,
                    f"Error updating '{metric.name}': {e}",
                    level="error"
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
        from django.shortcuts import redirect
        return redirect("admin:dashboard_projectmetricdata_add")

    def add_view(self, request, form_url="", extra_context=None):
        """
        Override the add view to show both single-value entry and CSV upload.
        """
        single_form = self.get_form(request)(request.POST or None)
        csv_form = utils.CSVUploadForm(request.POST or None, request.FILES or None)

        if request.method == "POST":
            if "csv_file" in request.FILES:  # CSV upload case
                if csv_form.is_valid():
                    return self.handle_csv_upload(request, csv_form)
            else:  # single entry case
                if single_form.is_valid():
                    obj = single_form.save(commit=False)
                    obj.save()
                    single_form.save_m2m()
                    self.save_related(request, single_form, [], False)

                    # 🔹 Update related ProjectMetrics
                    for metric in obj.project_metrics.all():
                        if metric.current_value is None:
                            delta_value = obj.value
                        else:
                            delta_value = obj.value - metric.current_value

                        # overwrite stored value with delta
                        obj.value = round(delta_value, 2)
                        obj.save(update_fields=["value"])

                        # update the metric itself
                        metric.current_value = round(obj.value + (metric.current_value or 0), 2)
                        metric.current_value_date = obj.date
                        metric.save(update_fields=["current_value", "current_value_date"])

                    messages.success(request, "Single ProjectMetricData record added.")
                    return redirect("admin:dashboard_projectmetricdata_add")

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "single_form": single_form,
            "csv_form": csv_form,
            "title": "Add Project Metric Data (single or bulk)",
        }
        return render(request, "admin/projectmetricdata_add.html", context)

    def handle_csv_upload(self, request, form):
        """
        Process uploaded CSV and create records for one metric.
        """
        csv_file = form.cleaned_data["csv_file"]
        metric = form.cleaned_data["project_metric"]

        decoded_file = csv_file.read().decode("utf-8-sig").splitlines()
        reader = csv.DictReader(decoded_file)
        print(reader.fieldnames)

        if not {"date", "value"}.issubset(reader.fieldnames):
            messages.error(request, "CSV must have 'date' and 'value' columns.")
            return redirect("admin:dashboard_projectmetricdata_add")

        for row in reader:
            try:
                date = row["date"]
                value = float(row["value"])
            except Exception as e:
                messages.error(request, f"Invalid row {row}: {e}")
                continue

            # Delta calculation
            if metric.current_value is None:
                delta_value = value
            else:
                delta_value = value - metric.current_value

            record = ProjectMetricData.objects.create(
                value=round(delta_value, 2),
                date=date,
            )
            record.project_metrics.add(metric)

            # Update ProjectMetric
            metric.current_value = round(value, 2)
            metric.current_value_date = date
            metric.save(update_fields=["current_value", "current_value_date"])

        messages.success(
            request,
            f"CSV uploaded successfully for {metric.name} ({', '.join(p.name for p in metric.projects.all())})."
        )
        return redirect("admin:dashboard_projectmetricdata_add")

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ("name", "key", "active", "created_at")
    list_filter = ("active",)
    search_fields = ("name", "key")
    readonly_fields = ("key", "created_at")