import requests, json, csv
from urllib.request import urlopen
from django import forms
from django.utils.html import format_html
from django.contrib import admin, messages
from django.core.exceptions import ValidationError
from .models import Project, AggregateMetric, ProjectMetric, ProjectMetricData, APIKey
from . import utils
from django.utils import timezone
from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe

API_URL = "https://api.carboncopy.news/projects"


class ProjectSelectForm(forms.ModelForm):
    project_selector = forms.ChoiceField(label="Select Project", required=True)
    impact_json_text = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 10,
                "placeholder": "{\n  \"impact_data\": [...]\n}"
            }
        ),
        required=False,
        label="Impact JSON (optional)",
        help_text="Paste the full impact JSON for this project. If provided, this will overwrite any existing JSON."
    )

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

        # If editing an existing project, make selector read-only
        if self.instance and self.instance.pk:
            current_name = self.instance.name or "(unknown)"
            current_id = self.instance.baserow_id or ""
            self.fields["project_selector"].choices = [(current_id, current_name)]
            self.fields["project_selector"].initial = current_id
            self.fields["project_selector"].disabled = True
            self.fields["project_selector"].help_text = "This project is already linked and cannot be changed."

        # If instance already has JSON, pre-fill textarea
        if self.instance and self.instance.impact_json:
            self.fields["impact_json_text"].initial = json.dumps(self.instance.impact_json, indent=2)

    def clean_impact_json_text(self):
        """Validate pasted JSON."""
        text = self.cleaned_data.get("impact_json_text", "").strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e.msg}")
        if not isinstance(parsed, dict):
            raise ValidationError("Impact JSON must be a JSON object.")
        return parsed


    def save(self, commit=True):
        selected_id = self.cleaned_data["project_selector"]
        impact_json = self.cleaned_data.get("impact_json_text")

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

         # Save JSON field if provided
        if impact_json is not None:
            self.instance.impact_json = impact_json

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
    save_delta = forms.BooleanField(
        required=False,
        initial=False,
        label="Save as cumulative value",
        help_text="If checked, the entered value will be treated as a cumulative value. "
                  "If unchecked, the entered value will be treated as a one-off value."
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
    list_display = ("name", "slug", "baserow_id", "logo_url", "last_updated")
    actions = ["update_impact_data", "refresh_baserow_data"]

    def refresh_baserow_data(self, request, queryset):
        """
        Re-fetches the latest project details (name, slug, logo, etc.)
        from the Baserow API for selected projects.
        """
        updated_count = 0
        failed_count = 0

        for project in queryset:
            try:
                response = utils.get_baserow_project_data(project.baserow_id)

                project.name = response.get("name", project.name)
                project.slug = response.get("slug", project.slug)
                project.logo_url = response.get("logo", project.logo_url)
                project.save(update_fields=["name", "slug", "logo_url", "last_updated"])

                updated_count += 1

            except requests.exceptions.RequestException as e:
                failed_count += 1
                self.message_user(
                    request,
                    f"Failed to refresh '{project.name}': {e}",
                    level=messages.ERROR,
                )

        if updated_count:
            self.message_user(
                request,
                f"Successfully refreshed {updated_count} project(s) from Baserow.",
                level=messages.SUCCESS,
            )

        if failed_count and not updated_count:
            self.message_user(
                request,
                f"All refresh attempts failed ({failed_count} projects).",
                level=messages.ERROR,
            )

    refresh_baserow_data.short_description = "Refresh project data"

    def update_impact_data(self, request, queryset):
        """
        Updates all metrics for each selected project using the stored impact_json.
        """
        results = []
        from . import update_metrics
        for project in queryset:
            try:
                if not project.impact_json:
                    self.message_user(
                        request,
                        f"Project '{project.name}' has no impact_json uploaded.",
                        level=messages.ERROR,
                    )
                    continue

                impact_data = project.impact_json.get("impact_data", [])

                if not isinstance(impact_data, list):
                    self.message_user(
                        request,
                        f"Invalid impact_json format for '{project.name}'",
                        level=messages.ERROR,
                    )
                    continue

                for item in impact_data:
                    source_value = item.get("source")
                    if not source_value:
                        continue

                    func_name = f"refresh_{source_value}"
                    func = getattr(update_metrics, func_name, None)

                    if not func:
                        self.message_user(
                            request,
                            f"Missing function '{func_name}' for project '{project.name}'",
                            level=messages.ERROR,
                        )
                        continue

                    results.extend(func(item))

                    if results is None:
                        self.message_user(
                            request,
                            f"Skipped '{project.name}' due to failed refresh ({source_value})",
                            level=messages.WARNING,
                        )
                        continue

                metrics = ProjectMetric.objects.filter(projects=project)
                updated_details = []  # <-- to store per-metric info

                for metric in metrics:
                    matched_metric = None

                    for result in results:
                        if not isinstance(result.value, (float, int)):
                            self.message_user(
                                request,
                                f"Invalid return value from {func_name} for '{metric.name}'",
                                level=messages.ERROR,
                            )
                            continue

                        if result.db_id == metric.db_id:
                            matched_metric = result
                            break

                    if not matched_metric:
                        continue

                    now = timezone.now()

                    last_record = (
                        ProjectMetricData.objects.filter(project_metrics=metric)
                        .order_by("-date")
                        .first()
                    )

                    if matched_metric.single == True:
                        record_value = matched_metric.value
                        metric.current_value = round(matched_metric.value + (metric.current_value or 0), 2)

                    else:
                        record_value = matched_metric.value - metric.current_value if last_record else matched_metric.value
                        metric.current_value = round(matched_metric.value, 2)
                        
                    record = ProjectMetricData.objects.create(
                        value=round(record_value, 2),
                        date=now,
                    )
                    record.project_metrics.add(metric)

                    metric.current_value_date = now 
                    metric.save(update_fields=["current_value", "current_value_date"])

                    updated_details.append(
                            f"{metric.name}: current={round(metric.current_value, 2)}, delta={round(record_value, 2)}"
                        )

                # Update project last_updated
                project.last_updated = timezone.now()
                project.save(update_fields=["last_updated"])

                if updated_details:
                    details_msg = "<br>".join(updated_details)
                    self.message_user(
                        request,
                        mark_safe(f"Updated metrics for '{project.name}':<br>{details_msg}"),
                        level=messages.SUCCESS,
                    )
                else:
                    self.message_user(
                        request,
                        f"No matching metrics updated for '{project.name}'.",
                        level=messages.WARNING,
                    )

            except Exception as e:
                self.message_user(
                    request,
                    f"Error updating project '{project.name}': {e}",
                    level=messages.ERROR,
                )

    update_impact_data.short_description = "Fetch impact data"


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

    def get_projects(self, obj):
        return ", ".join(p.name for p in obj.projects.all())
    
    get_projects.short_description = "Projects"

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
                    save_delta = single_form.cleaned_data.get("save_delta", False)

                    obj.save()
                    single_form.save_m2m()

                    for metric in obj.project_metrics.all():
                        if not save_delta:
                            # Case 1: Store inputted value in ProjectMetricData
                            delta_value = obj.value
                            metric.current_value = round(
                                (metric.current_value or 0) + delta_value, 2
                            )
                            obj.value = round(delta_value, 2)

                        else:
                            # Case 2: Store delta value in ProjectMetricData
                            if metric.current_value is None:
                                delta_value = obj.value
                            else:
                                delta_value = obj.value - metric.current_value

                            metric.current_value = round(obj.value, 2)
                            obj.value = round(delta_value, 2)

                        metric.current_value_date = obj.date
                        metric.save(update_fields=["current_value", "current_value_date"])
                        obj.save(update_fields=["value"])  # persist possibly updated value

                    messages.success(
                        request,
                        f"ProjectMetricData record added ({'delta' if save_delta else 'raw value'} mode)."
                    )
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