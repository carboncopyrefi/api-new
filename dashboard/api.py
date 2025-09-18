from fastapi import FastAPI, HTTPException, Depends
import django, os
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime, timedelta
from django.db.models import Sum, Max, Count
from django.db.models.functions import Coalesce, TruncMonth
from collections import defaultdict
from .utils import get_all_baserow_data
from .security import get_api_key
from django.db import connections

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
django.setup()

from .models import Project, ProjectMetric, ProjectMetricData as MetricData, AggregateMetric, AggregateMetricType  # noqa: E402

# app = FastAPI(title="CARBON Copy API", dependencies=[Depends(get_api_key)])
app = FastAPI(title="CARBON Copy API")

def get_django_db_connection():
    """
    FastAPI dependency to manage Django DB connection lifecycle.
    """ 
    conn = connections['default']

    def safe_is_usable(c):
        try:
            return c.is_usable()
        except Exception:
            return False
    try:
        if not safe_is_usable(conn):
            conn.close()
            conn.connect()
        yield conn
    finally:
        conn.close()

# -----------------------------
# Pydantic Models
# -----------------------------
class ProjectSummary(BaseModel):
    """Summary information for a project."""
    name: str = Field(..., example="Solar Energy Initiative")
    logo_url: Optional[str] = Field(None, example="https://example.com/logo.png")
    metrics: List[str] = Field(..., example=["Installed Capacity", "CO2 Savings"])

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
# Main Aggregator function
# ---------------------
def get_aggregate_metric_type_db_optimized(type_slug: str) -> AggregateMetricTypeResponse:
    """
    Returns the aggregate metric type with aggregated sums, percent changes,
    and a list of projects and their project metric values for each aggregate metric.
    """
    type_display = _get_type_display(type_slug)
    if not type_display:
        raise HTTPException(status_code=404, detail="Invalid aggregate metric type")

    agg_qs = AggregateMetric.objects.filter(type__slug=type_slug)
    
    metrics_out = []

    # Pre-build headers
    headers = ["Project Name"] + [
        f"{agg.name} ({agg.unit})" if agg.unit else agg.name
        for agg in agg_qs] + ["Last Updated"]

    # Build a map: {project_id: {metric_id: value}}
    project_metric_map = {}
    project_name_map = {}
    
    chart_metrics = agg_qs.filter(chart=True)  # only metrics that should have charts
    chart_data_map = defaultdict(lambda: {"month": None})  # keyed by month-year string

    for agg in agg_qs:
        pm_qs = ProjectMetric.objects.filter(aggregate_metric=agg)

        total_value_row = pm_qs.aggregate(total=Coalesce(Sum('current_value'), 0.0))
        total_value = float(total_value_row['total'] or 0.0)

        latest_date_row = pm_qs.aggregate(latest=Max('current_value_date'))
        latest_date = latest_date_row['latest']

        if latest_date is None:
            metrics_out.append(
                AggregateMetricItem(
                    name=agg.name,
                    value=total_value,
                    date=None,
                    unit=getattr(agg, 'unit', None),
                    format=getattr(agg, 'format', None),
                    description=getattr(agg, 'description', None),
                    percent_change_7d=None,
                    percent_change_28d=None,
                )
            )
            continue

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

        metrics_out.append(
            AggregateMetricItem(
                name=agg.name,
                value=total_value,
                date=latest_date.isoformat(),
                unit=getattr(agg, 'unit', None),
                format=getattr(agg, 'format', None),
                description=getattr(agg, 'description', None),
                percent_change_7d=percent_change_7d,
                percent_change_28d=percent_change_28d,
            )
        )

        # --- Fill project metric mapping ---
        for pm in pm_qs.prefetch_related('projects'):
            for project in pm.projects.all():
                project_id = project.id
                project_name_map[project_id] = project.name
                if project_id not in project_metric_map:
                    project_metric_map[project_id] = {}
                project_metric_map[project_id][agg.id] = pm.current_value

        # --- Chart Data ---
        if agg in chart_metrics:
            month_values = (
                MetricData.objects
                .filter(project_metrics__aggregate_metric=agg)
                .annotate(month=TruncMonth('date'))
                .values('month')
                .annotate(total=Coalesce(Sum('value'), 0.0))
                .order_by('month')
            )

            running_total = 0.0
            for entry in month_values:
                running_total += float(entry['total'] or 0.0)
                date_str = entry['month'].strftime('%Y-%m')
                chart_data_map[date_str]["month"] = date_str
                chart_data_map[date_str][agg.name] = running_total

        # ---- Pie chart (project breakdown) ----
        pie_chart_data = None

        # Which metric is flagged as the pie source?
        pie_metric = agg_qs.filter(pie_chart=True).first()

        # And is the TYPE configured for project-level pies?
        type_obj = AggregateMetricType.objects.only("pie_chart", "name").get(slug=type_slug)

        if pie_metric and type_obj.pie_chart == "project":
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
        type_name=type_display,
        description=type_obj.description,
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
    investment = get_aggregate_metric_type_db_optimized("investment")
    grants = get_aggregate_metric_type_db_optimized("grants")
    loans = get_aggregate_metric_type_db_optimized("lending")

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

@app.get("/", summary="Root endpoint")
def read_root():
    return {"message": "FastAPI + Django working"}


@app.get(
    "/projects",
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
                            "metrics": ["Installed Capacity", "CO2 Savings"]
                        },
                        {
                            "name": "Wind Farm Alpha",
                            "logo_url": "https://example.com/windfarm.png",
                            "metrics": ["Annual Output", "CO2 Savings"]
                        }
                    ]
                }
            }
        }
    }
)
def get_projects(db_conn=Depends(get_django_db_connection)):
    projects = Project.objects.all()
    return [
        ProjectSummary(
            name=project.name,
            logo_url=project.logo_url,
            metrics=list(
                project.metrics.values_list("name", flat=True).distinct()
            )
        )
        for project in projects
    ]


@app.get(
    "/projects/{baserow_id}/metrics",
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
    try:
        project = Project.objects.get(baserow_id=baserow_id)
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")

    def build_cumulative(metric):
        running_total = 0
        cumulative_data = []
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
            lambda cd: ProjectMetricData(
                name=metric.name,
                current_value=cd[-1][1] if cd else None,
                current_value_date=cd[-1][0] if cd else None,
                unit=metric.unit,
                format=metric.format,
                description=metric.description,
                percent_change_7d=(
                    ((cd[-1][1] - val_7d) / val_7d * 100)
                    if (val_7d := next(
                        (total for date, total in reversed(cd)
                         if date <= cd[-1][0] - timedelta(days=8)),
                        None
                    )) not in (None, 0)
                    else None
                ),
                percent_change_28d=(
                    ((cd[-1][1] - val_28d) / val_28d * 100)
                    if (val_28d := next(
                        (total for date, total in reversed(cd)
                         if date <= cd[-1][0] - timedelta(days=28)),
                        None
                    )) not in (None, 0)
                    else None
                ),
            )
        )(build_cumulative(metric))
        for metric in project.metrics.all()
    ]

@app.get(
    "/aggregate-metric-types",
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
    "/aggregate-metric-types/{type_slug}",
    response_model=AggregateMetricTypeResponse,
    summary="Get Aggregate Metric Type",
    description="Returns the aggregate metrics for the given type slug. The type slug must exist in AggregateMetric.TYPE_CHOICES."
)
def aggregate_metric_type_endpoint(type_slug: str):
    return get_aggregate_metric_type_db_optimized(type_slug)

@app.get(
    "/overview",
    response_model=OverviewResponse,
    summary="Overview of Funding Metrics"
)
def get_overview():
    return get_overview_data()

@app.get(
    "/venture-funding",
    response_model=VentureFundingResponse,
    summary="Venture Funding Overview",
    description="Returns total venture funding, deals, charts, project breakdown, and current year deals"
)
def venture_funding_endpoint():
    return get_venture_funding_data()

# @app.get("/test-db", summary="Test DB connection recovery")
# def test_db_connection():
#     from django.db import connections
#     from fastapi.responses import JSONResponse
#     conn = connections['default']

#     def safe_is_usable(c):
#         try:
#             return c.is_usable()
#         except Exception:
#             return False

#     # Step 1: Check current connection usability
#     before = safe_is_usable(conn)

#     # Step 2: Force-close the connection
#     conn.close()
#     forced_closed = not safe_is_usable(conn)

#     # Step 3: Run Django’s cleanup
#     close_old_connections()
#     after = safe_is_usable(connections['default'])

#     # Step 4: Try an actual query to prove recovery
#     from .models import Project
#     try:
#         count = Project.objects.count()
#         query_ok = True
#     except Exception as e:
#         count = str(e)
#         query_ok = False

#     return JSONResponse({
#         "before_close": before,
#         "after_forced_close": forced_closed,
#         "after_cleanup": after,
#         "query_ok": query_ok,
#         "project_count": count,
#     })