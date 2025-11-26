from fastapi import FastAPI, HTTPException, Depends
import django, os
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from django.db.models import Sum, Max, Count
from django.db.models.functions import Coalesce, TruncMonth
from collections import defaultdict
from .utils import get_all_baserow_data
from .security import get_api_key
from django.db import connections, transaction
from django.conf import settings
from django.utils.timezone import make_aware
from .middleware import DjangoDBMiddleware

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
django.setup()

from .models import Project, ProjectMetric, ProjectMetricData as MetricData, AggregateMetric, SDG, AggregateMetricType  # noqa: E402

# app = FastAPI(title="CARBON Copy API", dependencies=[Depends(get_api_key)])
app = FastAPI(title="CARBON Copy API")
app.add_middleware(DjangoDBMiddleware)

# -----------------------------
# Pydantic Models
# -----------------------------
class ProjectSummary(BaseModel):
    """Summary information for a project."""
    name: str = Field(..., example="Solar Energy Initiative")
    logo_url: Optional[str] = Field(None, example="https://example.com/logo.png")
    metrics: List[str] = Field(..., example=["Installed Capacity", "CO2 Savings"])
    slug: Optional[str] = Field(None, example="solar-energy-initiative")

class ProjectMetricData(BaseModel):
    """Detailed information about a project metric."""
    name: str = Field(..., example="Installed Capacity")
    current_value: Optional[float] = Field(None, example=25.4)
    current_value_date: Optional[datetime] = Field(None, example="2025-08-01T14:30:00Z")
    unit: Optional[str] = Field(None, example="MW")
    format: Optional[str] = Field(None, description="Display format for the metric value", example="number")
    description: Optional[str] = Field(None, example="Total installed renewable energy capacity in megawatts")
    percent_change_7d: Optional[float] = None
    percent_change_28d: Optional[float] = None

class AggregateMetricTypeList(BaseModel):
    name: str = Field(..., example="Total Installed Capacity")
    description: Optional[str] = Field(None, example="Sum of installed capacity across all projects")   
    slug: str = Field(..., example="total-installed-capacity")
    pie_chart: str = Field(..., example="Project", description="Pie chart grouping for this metric type")

class AggregateMetricItem(BaseModel):
    name: str
    value: float = Field(..., description="Sum of current_value across project metrics")
    date: Optional[str] = Field(None, description="ISO timestamp of latest underlying metric date")
    unit: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    percent_change_7d: Optional[float] = Field(None, description="Percent change vs ~7 days ago")
    percent_change_28d: Optional[float] = Field(None, description="Percent change vs ~28 days ago")

class AggregateMetricTypeTable(BaseModel):
    headers: List[str]
    rows: List[List[Union[str, float, None]]]

class SDGList(BaseModel):
    name: str = Field(..., example="Goal #1 - No Poverty")
    description: Optional[str] = Field(None, example="SDG description")   
    slug: str = Field(..., example="1-no-poverty")
    metrics: List[AggregateMetricItem]

class PieChartDataItem(BaseModel):
    name: str  # project name
    value: float  # sum of metric values for this project
    project_id: Optional[int] = None  # optional, for linking on frontend

class PieChartData(BaseModel):
    title: str
    items: List[PieChartDataItem]

class AggregateMetricTypeResponse(BaseModel):
    type_name: str
    description: Optional[str] = None
    metrics: List[AggregateMetricItem]
    table: AggregateMetricTypeTable
    projects_count: int = Field(..., description="Number of distinct projects contributing to this metric type")
    charts: Optional[List[dict]] = Field(None, description="Chart data for Recharts visualization")
    pie_chart: Optional[PieChartData] = Field(
        None,
        description="Pie chart breakdown for metrics flagged with pie_chart=True"
    )

class OverviewMetric(BaseModel):
    current: float
    change7d: Optional[float] = None
    change28d: Optional[float] = None

class OverviewResponse(BaseModel):
    investment: OverviewMetric
    grants: OverviewMetric
    loans: OverviewMetric
    total: OverviewMetric
    timeseries: List[dict]

class VentureFundingMetrics(BaseModel):
    total_funding: float
    total_deals: int

class VentureFundingChartPoint(BaseModel):
    x: str  # month-year
    y: float

class VentureFundingProject(BaseModel):
    name: str
    total_funding: float
    deal_count: int

class VentureFundingDeal(BaseModel):
    project: str
    amount: float

class VentureFundingResponse(BaseModel):
    metrics: VentureFundingMetrics
    charts: dict  # {"funding_by_month": [...], "deals_by_month": [...]}
    projects: List[VentureFundingProject]
    current_year_deals: List[VentureFundingDeal]

# ---------------------
# Helper: validate type_slug
# ---------------------
def _get_type_display(type_slug: str) -> Optional[str]:
    """Return the display name (name) for a type_slug or None if invalid."""
    try:
        return AggregateMetricType.objects.get(slug=type_slug).name
    except AggregateMetricType.DoesNotExist:
        return None
    
# ---------------------
# Helper: validate sdg_slug
# ---------------------
def _get_sdg_display(sdg_slug: str) -> Optional[str]:
    """Return the display name (name) for a sdg_slug or None if invalid."""
    try:
        return SDG.objects.get(slug=sdg_slug).name
    except SDG.DoesNotExist:
        return None

# ---------------------
# Global aggregate metric value and percent calculator
# ---------------------
def _agg_metric_calc(agg: str):
    pm_qs = ProjectMetric.objects.filter(aggregate_metric=agg)

    total_value_row = pm_qs.aggregate(total=Coalesce(Sum('current_value'), 0.0))
    total_value = float(total_value_row['total'] or 0.0)

    naive_latest_date = datetime.now()
    settings.TIME_ZONE
    latest_date = make_aware(naive_latest_date)

    if latest_date is None:
        return AggregateMetricItem(
            name=agg.name,
            value=total_value,
            date=None,
            unit=getattr(agg, 'unit', None),
            format=getattr(agg, 'format', None),
            description=getattr(agg, 'description', None),
            percent_change_7d=None,
            percent_change_28d=None,
        ), pm_qs

    target_7d = latest_date - timedelta(days=8)
    target_28d = latest_date - timedelta(days=28)

    prev7_total = MetricData.objects.filter(
        project_metrics__aggregate_metric=agg,
        date__lte=target_7d
    ).aggregate(total=Coalesce(Sum('value'), 0.0))['total'] or 0.0

    prev28_total = MetricData.objects.filter(
        project_metrics__aggregate_metric=agg,
        date__lte=target_28d
    ).aggregate(total=Coalesce(Sum('value'), 0.0))['total'] or 0.0

    percent_change_7d = None
    if prev7_total not in (0, None):
        percent_change_7d = (total_value - prev7_total) / prev7_total * 100.0

    percent_change_28d = None
    if prev28_total not in (0, None):
        percent_change_28d = (total_value - prev28_total) / prev28_total * 100.0

    return AggregateMetricItem(
        name=agg.name,
        value=total_value,
        date=latest_date.isoformat(),
        unit=getattr(agg, 'unit', None),
        format=getattr(agg, 'format', None),
        description=getattr(agg, 'description', None),
        percent_change_7d=percent_change_7d,
        percent_change_28d=percent_change_28d,
    ), pm_qs


# ---------------------
# Main Aggregator function
# ---------------------
def get_aggregate_metric_type_db_optimized(slug: str, slug_type: str) -> AggregateMetricTypeResponse:
    """
    Returns the aggregate metric type with aggregated sums, percent changes,
    and a list of projects and their project metric values for each aggregate metric.
    """
    if slug_type == "sdg":
        display = _get_sdg_display(slug)
        if not display:
            raise HTTPException(status_code=404, detail="Invalid SDG")
        
        agg_qs = AggregateMetric.objects.filter(sdg__slug=slug)
    
    else:
        display = _get_type_display(slug)
        if not display:
            raise HTTPException(status_code=404, detail="Invalid aggregate metric type")

        agg_qs = AggregateMetric.objects.filter(type__slug=slug)
    
    metrics_out = []

    # Pre-build headers
    headers = ["Project Name"] + [
        f"{agg.name} ({agg.unit})" if agg.unit else agg.name
        for agg in agg_qs] + ["Last Updated"]

    # Build a map: {project_id: {metric_id: value}}
    project_metric_map = {}
    project_name_map = {}
    
    chart_metrics = agg_qs.filter(chart=True)  # only metrics that should have charts
    chart_data_map = defaultdict(dict)

    for agg in agg_qs:
        
        # --- Run metric calculations and percent change ---
        agg_item, pm_qs = _agg_metric_calc(agg)
        metrics_out.append(agg_item)

        # --- Fill project metric mapping ---
        for pm in pm_qs.prefetch_related('projects'):
            for project in pm.projects.all():
                project_id = project.id
                project_name_map[project_id] = project.name
                if project_id not in project_metric_map:
                    project_metric_map[project_id] = {}
                project_metric_map[project_id][agg.id] = pm.current_value

        # --- Chart Data ---
        # 1. Collect month ranges across all metrics
        all_months = set()
        metric_month_maps = {}

        for agg in chart_metrics:
            month_values = (
                MetricData.objects
                .filter(project_metrics__aggregate_metric=agg)
                .annotate(month=TruncMonth('date'))
                .values('month')
                .annotate(total=Coalesce(Sum('value'), 0.0))
                .order_by('month')
            )

            # Store per-metric data
            raw_month_map = {
                entry['month'].strftime('%Y-%m'): float(entry['total'] or 0.0)
                for entry in month_values
            }
            metric_month_maps[agg.name] = raw_month_map

            # Add these months into global axis
            for entry in month_values:
                all_months.add(entry['month'])

        # 2. Build full continuous month axis (min → max)
        if all_months:
            min_month = min(all_months)
            max_month = max(all_months)

            current = min_month
            month_axis = []
            while current <= max_month:
                month_axis.append(current.strftime('%Y-%m'))
                current += relativedelta(months=1)

        # 3. Fill data for each metric using running totals
        for agg in chart_metrics:
            raw_month_map = metric_month_maps.get(agg.name, {})
            running_total = 0.0
            last_value = 0.0

            for date_str in month_axis:
                chart_data_map[date_str]["month"] = date_str

                if date_str in raw_month_map:
                    running_total += raw_month_map[date_str]
                    last_value = running_total

                chart_data_map[date_str][agg.name] = last_value

        # 4. Convert chart_data_map to sorted list
        chart_data = [chart_data_map[m] for m in sorted(chart_data_map)]

        # ---- Pie chart (project breakdown) ----
        pie_chart_data = None

        # Which metric is flagged as the pie source?
        pie_metric = agg_qs.filter(pie_chart=True).first()

        # And is the TYPE configured for project-level pies?
        if slug_type == "sdg":
            obj = SDG.objects.only("name").get(slug=slug)
            pie_chart_items = None
        else:
            obj = AggregateMetricType.objects.only("pie_chart", "name").get(slug=slug)

            if pie_metric and obj.pie_chart == "project":
                pie_chart_items = []
                for project_id, metrics_dict in project_metric_map.items():
                    value = metrics_dict.get(pie_metric.id, 0)
                    if value not in (None, 0):
                        pie_chart_items.append({
                            "name": project_name_map[project_id],
                            "value": float(value),
                            "project_id": project_id,
                        })

                if pie_chart_items:
                    pie_chart_data = {
                        "title": pie_metric.name,
                        "items": pie_chart_items
                    }

    # Build rows
    rows = []
    for project_id, metrics_dict in project_metric_map.items():
        project = Project.objects.get(id=project_id)

        # Find latest current_value_date across all metrics for this project
        latest_date = (
            project.metrics.aggregate(
                latest=Max("current_value_date")
            )["latest"]
        )

        row = [project_name_map[project_id]]
        for agg in agg_qs:
            value = metrics_dict.get(agg.id, None)
            row.append(value)

        # Append Last Updated as the final column
        row.append(latest_date.date().isoformat() if latest_date else None)
        rows.append(row)

    table = AggregateMetricTypeTable(headers=headers, rows=rows)

    # Calculate distinct projects count
    projects_count = Project.objects.filter(
        metrics__aggregate_metric__in=agg_qs
    ).distinct().count()

    # Convert chart data map to sorted list for Recharts
    chart_data = sorted(chart_data_map.values(), key=lambda x: x['month'])

    return AggregateMetricTypeResponse(
        type_name=display,
        description=obj.description,
        projects_count=projects_count,
        metrics=metrics_out,
        table=table,
        charts=chart_data if chart_data else None,
        pie_chart=pie_chart_data if pie_chart_items else None,
    )

# -----------------------------
# Overview page function
# -----------------------------
def get_overview_data() -> OverviewResponse:
    # Fetch three types
    with transaction.atomic():
        investment = get_aggregate_metric_type_db_optimized("investment", "type")
        grants = get_aggregate_metric_type_db_optimized("grants", "type")
        loans = get_aggregate_metric_type_db_optimized("lending", "type")

    def extract(metric_resp: AggregateMetricTypeResponse) -> OverviewMetric:
        # Use the first metric (assuming each type only has one top-level aggregate)
        m = metric_resp.metrics[0]
        return OverviewMetric(
            current=m.value,
            change7d=round(m.percent_change_7d,2),
            change28d=round(m.percent_change_28d,2),
        )

    inv = extract(investment)
    gr = extract(grants)
    ln = extract(loans)

    total_current = (inv.current or 0) + (gr.current or 0) + (ln.current or 0)

    def safe_avg(*vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    total = OverviewMetric(
        current=total_current,
        change7d=round(safe_avg(inv.change7d, gr.change7d, ln.change7d),2),
        change28d=round(safe_avg(inv.change28d, gr.change28d, ln.change28d),2),
    )

    # Build combined timeseries using the three metrics
    # (for now just use investment.timeseries as example)
    # Later, you might want to sum across them at each date
    timeseries = []
    inv_ts = {e["month"]: e["Investment in Impact Projects"] for e in investment.charts or []}
    gr_ts = {e["month"]: e["Granted to Impact Projects"] for e in grants.charts or []}
    ln_ts = {e["month"]: e["Lent to Impact Projects"] for e in loans.charts or []}

    all_months = sorted(set(inv_ts) | set(gr_ts) | set(ln_ts))

    cumulative = 0
    prev_inv = prev_gr = prev_ln = 0

    for m in all_months:
        # snapshot values at month m
        inv_val = inv_ts.get(m, prev_inv)
        gr_val = gr_ts.get(m, prev_gr)
        ln_val = ln_ts.get(m, prev_ln)

        # deltas for each series
        delta_inv = inv_val - prev_inv
        delta_gr = gr_val - prev_gr
        delta_ln = ln_val - prev_ln

        # update cumulative
        cumulative += delta_inv + delta_gr + delta_ln

        # update previous values
        prev_inv, prev_gr, prev_ln = inv_val, gr_val, ln_val

        timeseries.append({
            "date": m,
            "Total Funding to Impact Projects": cumulative
        })

    return OverviewResponse(
        investment=inv,
        grants=gr,
        loans=ln,
        total=total,
        timeseries=timeseries
    )

# -----------------------------
# Venture Funding function
# -----------------------------
def get_venture_funding_data() -> VentureFundingResponse:
    page_size = 200
    params = (
        f"size={page_size}"
        f"&order_by=-Date"
        f"&filter__field_2209786__single_select_is_any_of=1686865,1688192"
    )

    records = get_all_baserow_data("306630", params)

    total_funding = 0
    total_deals = len(records)

    funding_by_year = defaultdict(float)
    deals_by_year = defaultdict(int)
    project_funding = defaultdict(lambda: {"total": 0, "count": 0})
    current_year_deals = []

    this_year = datetime.now().year

    for r in records:
        amount = float(r["Amount"])
        project = r["Company"][0]["value"] if r.get("Company") else "Unknown"
        date = datetime.strptime(r["Date"], "%Y-%m-%d")
        year_key = str(date.year)

        total_funding += amount
        funding_by_year[year_key] += amount
        deals_by_year[year_key] += 1

        project_funding[project]["total"] += amount
        project_funding[project]["count"] += 1

        if date.year == this_year:
            current_year_deals.append(
                VentureFundingDeal(project=project, amount=amount)
            )

    return VentureFundingResponse(
        metrics=VentureFundingMetrics(
            total_funding=total_funding,
            total_deals=total_deals,
        ),
        charts={
            "funding_by_year": [
                {"x": k, "y": v} for k, v in sorted(funding_by_year.items())
            ],
            "deals_by_year": [
                {"x": k, "y": v} for k, v in sorted(deals_by_year.items())
            ],
        },
        projects=sorted(
            [
                VentureFundingProject(
                    name=name,
                    total_funding=pf["total"],
                    deal_count=pf["count"],
                )
                for name, pf in project_funding.items()
            ],
            key=lambda p: p.total_funding,
            reverse=True,  # largest funding first
        ),
        current_year_deals=current_year_deals,
    )

# -----------------------------
# Routes
# -----------------------------

@app.get("/", summary="Root endpoint", dependencies=[])
def read_root():
    return {
        "message": "CARBON Copy API. See the documentation at /docs",
        "docs_url": "/docs"
    }


@app.get(
    "/projects",
    dependencies=[Depends(get_api_key)],
    response_model=List[ProjectSummary],
    summary="List all projects",
    responses={
        200: {
            "description": "List of projects with basic info",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "Solar Energy Initiative",
                            "logo_url": "https://example.com/logo.png",
                            "metrics": ["Installed Capacity", "CO2 Savings"],
                            "slug": "solar-energy-initiative"
                        },
                        {
                            "name": "Wind Farm Alpha",
                            "logo_url": "https://example.com/windfarm.png",
                            "metrics": ["Annual Output", "CO2 Savings"],
                            "slug": "wind-farm-alpha"
                        }
                    ]
                }
            }
        }
    }
)
def get_projects():
    with transaction.atomic():
        projects = Project.objects.all().order_by("name")
    return [
        ProjectSummary(
            name=project.name,
            logo_url=project.logo_url,
            metrics=list(
                project.metrics.values_list("name", flat=True).distinct()
            ),
            slug=project.slug
        )
        for project in projects
    ]


@app.get(
    "/projects/{baserow_id}/metrics",
    dependencies=[Depends(get_api_key)],
    response_model=List[ProjectMetricData],
    summary="Get metrics for a specific project",
    responses={
        200: {
            "description": "List of metrics for the given project",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "Installed Capacity",
                            "current_value": 25.4,
                            "current_value_date": "2025-08-01T14:30:00Z",
                            "unit": "MW",
                            "format": "number",
                            "description": "Total installed renewable energy capacity in megawatts"
                        },
                        {
                            "name": "CO2 Savings",
                            "current_value": 1500.75,
                            "current_value_date": "2025-08-01T14:30:00Z",
                            "unit": "tCO2",
                            "format": "number",
                            "description": "Estimated CO2 emissions avoided per year"
                        }
                    ]
                }
            }
        },
        404: {"description": "Project not found"}
    }
)
def get_project_metrics_data(baserow_id: int):
    naive_latest_date = datetime.now()
    settings.TIME_ZONE
    today = make_aware(naive_latest_date)

    try:
        with transaction.atomic():
            project = Project.objects.get(baserow_id=baserow_id)
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")

    def build_cumulative(metric):
        running_total = 0
        cumulative_data = []
        with transaction.atomic():
            for date, value in (
                MetricData.objects
                .filter(project_metrics=metric)
                .order_by("date")
                .values_list("date", "value")
            ):
                running_total += float(value or 0)
                cumulative_data.append((date, running_total))
        return cumulative_data

    return [
    (
        lambda cd: (
            lambda latest_date: ProjectMetricData(
                name=metric.name,
                current_value=cd[-1][1] if cd else None,
                current_value_date=cd[-1][0] if cd else None,
                unit=metric.unit,
                format=metric.format,
                description=metric.description,
                percent_change_7d=(
                    round(((cd[-1][1] - val_7d) / val_7d * 100), 2)
                    if cd and (val_7d := next(
                        iter([total for date, total in reversed(cd) if date <= today - timedelta(days=7)]),
                        None
                    )) not in (None, 0)
                    else None
                ),
                percent_change_28d=(
                    round(((cd[-1][1] - val_28d) / val_28d * 100), 2)
                    if cd and (val_28d := next(
                        iter([total for date, total in reversed(cd) if date <= today - timedelta(days=28)]),
                        None
                    )) not in (None, 0)
                    else None
                ),
            )
        )(datetime.now().date())
    )(build_cumulative(metric))
    for metric in project.metrics.all()
]

@app.get(
    "/aggregate-metric-types",
    dependencies=[Depends(get_api_key)],
    response_model=List[AggregateMetricTypeList],
    summary="List all aggregate metric types",
    responses={
        200: {
            "description": "List of aggregate metric types with slugs",
            "content": {
                "application/json": {
                    "example": [
                        {"name": "Ecological Credits", "slug": "ecological-credits"},
                        {"name": "Waste", "slug": "waste"}
                    ]
                }
            }
        }
    }
)
def get_aggregate_metric_types():
    from .models import AggregateMetricType
    with transaction.atomic():
        types = AggregateMetricType.objects.all().order_by("name")
    return [
        AggregateMetricTypeList(
            name=t.name,
            description=t.description,
            slug=t.slug,
            pie_chart=t.pie_chart
        )
        for t in types
    ]

@app.get(
    "/aggregate-metric-types/{slug}",
    dependencies=[Depends(get_api_key)],
    response_model=AggregateMetricTypeResponse,
    summary="Get aggregate metrics by Type",
    description="Returns the aggregate metrics for the given type slug. The type slug must exist in Aggregate Metric Type table."
)
def aggregate_metric_type_endpoint(slug: str):
    with transaction.atomic():
        data = get_aggregate_metric_type_db_optimized(slug, "type")
    return data

@app.get(
    "/sdg",
    dependencies=[],
    response_model=List[SDGList],
    summary="List all SDGs",
    responses={
        200: {
            "description": "List of SDGs with slugs",
            "content": {
                "application/json": {
                    "example": [
                        {"name": "Goal #1 - No Poverty", "description": "SDG description", "slug": "1-no-poverty"},
                        {"name": "Goal #2 - Zero Hunger", "description": "SDG description", "slug": "2-zero-hunger"}
                    ]
                }
            }
        }
    }
)
def get_aggregate_metric_types():
    sdgs = SDG.objects.all()
    sdg_out = []

    for sdg in sdgs:
        agg_qs = AggregateMetric.objects.filter(sdg=sdg)

        metrics_out = []
        for agg in agg_qs:
            data = _agg_metric_calc(agg)
            metrics_out.append(data[0])  # AggregateMetricItem

        sdg_out.append(
            SDGList(
                name=sdg.name,
                description=sdg.description,
                slug=sdg.slug,
                metrics=metrics_out,
            )
        )

    return sdg_out

@app.get(
    "/sdg/{slug}",
    dependencies=[],
    response_model=AggregateMetricTypeResponse,
    summary="Get aggregate metric data by SDG",
    description="Returns the aggregate metrics for the given SDG slug. The SDG slug must exist in the SDG table."
)
def aggregate_metric_type_endpoint(slug: str):
    with transaction.atomic():
        data = get_aggregate_metric_type_db_optimized(slug, "sdg")
    return data

@app.get(
    "/overview",
    response_model=OverviewResponse,
    summary="Overview of Funding Metrics"
)
def get_overview():
    return get_overview_data()

@app.get(
    "/venture-funding",
    dependencies=[Depends(get_api_key)],
    response_model=VentureFundingResponse,
    summary="Venture Funding Overview",
    description="Returns total venture funding, deals, charts, project breakdown, and current year deals"
)
def venture_funding_endpoint():
    return get_venture_funding_data()


@app.get("/link-preview", dependencies=[Depends(get_api_key)])
async def link_preview(url: str):
    import requests
    from bs4 import BeautifulSoup

    res = requests.get(url, timeout=5)
    soup = BeautifulSoup(res.text, "html.parser")

    def get_meta(prop):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return tag["content"] if tag and "content" in tag.attrs else ""

    return {
        "title": get_meta("og:title") or soup.title.string if soup.title else url,
        "description": get_meta("og:description"),
        "image": get_meta("og:image"),
    }
