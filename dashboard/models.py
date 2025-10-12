from django.db import models
import secrets

format_choices = [
        ('{:,.2f}', 'Comma separated, 2 decimal places'),
        ('{:,.0f}', 'Comma separated, no decimals'),
    ]

pie_chart_choices = [
        ('project', 'Project'),
        ('category', 'Category'),
        ('chain', 'Chain'),
    ]

class Project(models.Model):
    baserow_id = models.PositiveIntegerField("Baserow ID", unique=True)
    name = models.CharField(max_length=255)
    logo_url = models.URLField(blank=True, null=True)
    slug = models.CharField(max_length=100, null=True)
    impact_json = models.JSONField(null=True, blank=True, help_text="Optional Impact Metrics JSON for this project")
    last_updated = models.DateTimeField(blank=True, null=True) 

    def __str__(self):
        return self.name

class AggregateMetricType(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=200, null=True, blank=True)
    slug = models.SlugField(unique=True)
    pie_chart = models.CharField(max_length=20, choices=pie_chart_choices)

    def __str__(self):
        return self.name
    
class AggregateMetric(models.Model):
    name = models.CharField(max_length=255)
    unit = models.CharField(max_length=50, blank=True, null=True)
    format = models.CharField(max_length=10, choices=format_choices)
    description = models.CharField(max_length=500)
    chart = models.BooleanField(default=False)
    pie_chart = models.BooleanField(default=False)

    type = models.ForeignKey(
        AggregateMetricType,
        on_delete=models.CASCADE,
        related_name="metrics"
    )

    def __str__(self):
        return self.name
    
class ProjectMetric(models.Model):
    CATEGORY_CHOICES = [
        ('plastic', 'Plastic'),
        ('solar', 'Solar'),
        ('carbon', 'Carbon'),
        ('biodiversity', 'Biodiversity'),
        ('cookstove', 'Cookstove'),
    ]

    db_id = models.IntegerField(
        verbose_name="DB ID",
        blank=True,
        null=True,
        unique=True
    )

    name = models.CharField(max_length=255)
    unit = models.CharField(max_length=50, blank=True, null=True)
    format = models.CharField(max_length=10, choices=format_choices)
    description = models.CharField(max_length=500)

    current_value = models.FloatField(blank=True, null=True)
    current_value_date = models.DateTimeField(blank=True, null=True)

    category = models.CharField(
        max_length=20, choices=CATEGORY_CHOICES, blank=True, null=True
    )
    projects = models.ManyToManyField(
        'Project',
        related_name="metrics",
        blank=False
    )
    aggregate_metric = models.ForeignKey(
        'AggregateMetric', on_delete=models.SET_NULL, blank=True, null=True
    )

    def __str__(self):
        return self.name
    
class ProjectMetricData(models.Model):
    value = models.FloatField()
    date = models.DateTimeField(db_index=True)
    project_metrics = models.ManyToManyField(
        'ProjectMetric',
        related_name='metric_data'
    )

    class Meta:
        verbose_name = "Project Metric Data"
        verbose_name_plural = "Project Metric Data"
        indexes = [
            models.Index(fields=['date']),
        ]

    def __str__(self):
        return f"{self.value} on {self.date}"
    

class APIKey(models.Model):
    name = models.CharField(max_length=100, help_text="Label for the key (e.g. client name)")
    key = models.CharField(max_length=64, unique=True, db_index=True, editable=False)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @staticmethod
    def generate_key():
        # Generates a 64-char hex string
        return secrets.token_hex(32)

    def save(self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_key()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({'active' if self.active else 'inactive'})"