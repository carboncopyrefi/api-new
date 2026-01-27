from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import django, os, json, requests, feedparser, re, markdown
from typing import List, Optional, Union
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from django.db.models import Sum, Max, Count
from django.db.models.functions import Coalesce, TruncMonth
from collections import defaultdict
from .utils import (
    get_all_baserow_data,
    parse_datetime,
    get_giveth_data,
    calculate_dict_sums,
    get_coingeckoterminal_data,
    get_coingecko_data,
    get_karma_gap_data
)
from .security import get_api_key
from django.db import connections, transaction
from django.conf import settings
from django.utils.timezone import make_aware
from .middleware import DjangoDBMiddleware
from .schemas import (
    ImpactProjectSummary,
    ProjectSummary,
    Article,
    Token,
    Activity,
    CategoryResponse,
    NewsItem,
    ProjectMetricData,
    AggregateMetricTypeList,
    AggregateMetricItem,
    AggregateMetricTypeTable,
    SDGDetailResponse,
    SDGMetricGroup,
    SDGList,
    AggregateMetricTypeResponse,
    OverviewResponse,
    OverviewMetric,
    VentureFundingResponse,
    VentureFundingMetrics,
    VentureFundingProject,
    VentureFundingDeal,
    )

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
django.setup()

from .models import Project, ProjectMetric, ProjectMetricData as MetricData, AggregateMetric, SDG, AggregateMetricType, APIKey  # noqa: E402

# app = FastAPI(title="CARBON Copy API", dependencies=[Depends(get_api_key)])
app = FastAPI(title="CARBON Copy API")
app.add_middleware(DjangoDBMiddleware)

templates = Jinja2Templates(directory="templates")

# Create a limiter — identify clients by IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
# Helper: table builder
# ---------------------
def _build_table(aggs, project_metric_map, project_name_map):
    headers = ["Project Name"] + [
        f"{agg.name} ({agg.unit})" if agg.unit else agg.name
        for agg in aggs
    ] + ["Last Updated"]

    rows = []

    for project_id, metrics_dict in project_metric_map.items():
        project = Project.objects.get(id=project_id)

        latest_date = (
            project.metrics.aggregate(
                latest=Max("current_value_date")
            )["latest"]
        )

        row = [project_name_map[project_id]]
        for agg in aggs:
            row.append(metrics_dict.get(agg.id))
        row.append(latest_date.date().isoformat() if latest_date else None)

        rows.append(row)

    return AggregateMetricTypeTable(headers=headers, rows=rows)

# ---------------------
# Helper: chart builder
# ---------------------
def _build_charts(aggs, chart_flag_field):
    chart_metrics = [agg for agg in aggs if getattr(agg, chart_flag_field)]

    if not chart_metrics:
        return None

    all_months = set()
    metric_month_maps = {}

    for agg in chart_metrics:
        month_values = (
            MetricData.objects
            .filter(project_metrics__aggregate_metric=agg)
            .annotate(month=TruncMonth("date"))
            .values("month")
            .annotate(total=Coalesce(Sum("value"), 0.0))
            .order_by("month")
        )

        raw = {
            entry["month"].strftime("%Y-%m"): float(entry["total"] or 0.0)
            for entry in month_values
        }
        metric_month_maps[agg.name] = raw
        all_months.update(entry["month"] for entry in month_values)

    if not all_months:
        return None

    current = min(all_months)
    max_month = max(all_months)

    month_axis = []
    while current <= max_month:
        month_axis.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    chart_data_map = defaultdict(dict)

    for agg in chart_metrics:
        raw = metric_month_maps.get(agg.name, {})
        running = 0.0

        for month in month_axis:
            chart_data_map[month]["month"] = month
            if month in raw:
                running += raw[month]
            chart_data_map[month][agg.name] = running

    return sorted(chart_data_map.values(), key=lambda x: x["month"])

# ---------------------
# Helper: pie chart builder
# ---------------------
def _build_pie(
    aggs,
    project_metric_map,
    project_name_map,
    *,
    metric_type,
):
    """
    Build pie chart data for an aggregate metric type.
    Returns None if no pie chart applies.
    """

    if metric_type.pie_chart != "project":
        return None

    pie_metric = next(
        (agg for agg in aggs if agg.pie_chart),
        None
    )

    if not pie_metric:
        return None

    items = []

    for project_id, metrics_dict in project_metric_map.items():
        value = metrics_dict.get(pie_metric.id)
        if value not in (None, 0):
            items.append({
                "name": project_name_map[project_id],
                "value": float(value),
                "project_id": project_id,
            })

    if not items:
        return None

    return {
        "title": pie_metric.name,
        "items": items,
    }

# ---------------------
# Helper: metric group builder
# ---------------------
def _build_metric_group(
    aggs,
    *,
    chart_flag_field,
    include_pie = None,
    metric_type=None,
):
    if include_pie and not metric_type:
        raise ValueError("metric_type is required when include_pie=True")

    metrics_out = []
    project_metric_map = {}
    project_name_map = {}

    for agg in aggs:
        agg_item, pm_qs = _agg_metric_calc(agg)
        metrics_out.append(agg_item)

        for pm in pm_qs.prefetch_related("projects"):
            for project in pm.projects.all():
                pid = project.id
                project_name_map[pid] = project.name
                project_metric_map.setdefault(pid, {})[agg.id] = pm.current_value

    table = _build_table(aggs, project_metric_map, project_name_map)
    charts = _build_charts(aggs, chart_flag_field)

    pie_chart = None
    if include_pie:
        pie_chart = _build_pie(
            aggs,
            project_metric_map,
            project_name_map,
            metric_type=metric_type,
        )

    return {
        "metrics": metrics_out,
        "table": table,
        "charts": charts,
        "pie_chart": pie_chart,
    }

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

    records = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY_FUNDRAISING"), params)

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
# Project dynamic content function
# -----------------------------

def dynamic_project_content(slug):
    content = {}
    
    params = "filter__field_1248804__equal=" + slug
    try:
        data = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY"), params)
    except Exception:
        raise HTTPException(status_code=500, detail="Error getting project details from Baserow")
    
    if len(data) < 1:
        raise HTTPException(status_code=404, detail="Project not found")
    else:
        result = data[0]
        company_id = str(result['id'])
        company_name = result['Name']

    try:
        # RSS feed content
        generator = "" 

        if result['Content feed'] == "":
            content_list = None        
        else:
            content_feed_url = str(result['Content feed'])
            article_list = []
            content_list = []
            mainImage = "" 

            if "paragraph" in content_feed_url:
                r = requests.get(content_feed_url)
                f = feedparser.parse(r.text)
            else:
                f = feedparser.parse(content_feed_url)
                
            if hasattr(f.feed,'generator'): 
                if f.feed['generator'] == 'Medium':
                    generator = 'Medium'
            else:
                generator = None

            for article in f.entries[0:3]:
                if hasattr(f.feed, 'image'):
                    mainImage = f.feed['image']['href']
                link = ""
                date = parse_datetime(article.published)
                formatted_date = date.strftime(os.getenv("DATE_FORMAT"))
                if generator == 'Medium':
                    match = re.search(r'<img[^>]+src="([^">]+)"', article.content[0]['value'])
                    mainImage = match.group(1)
                if hasattr(article, 'media_content'):
                    mainImage = article.media_content[0]['url']
                if hasattr(article,'image'):
                    mainImage = article.image['href']
                if hasattr(article, 'links'):
                    for link in article.links:
                        if link.type == "image/jpg" or link.type == "image/jpeg":
                            mainImage = link.href
                        if link.type == 'audio/mpeg':
                            link = link.href
                        else:
                            link = article.link
                            continue
                else:
                    continue

                a = Article(
                    title=article.title,
                    url=link,
                    mainImage=mainImage,
                    publication=None,
                    date=formatted_date
                )
                article_list.append(a)

            for item in article_list:
                item_dict = vars(item)
                content_list.append(item_dict)
        content["content"] = content_list
    except Exception as e:
        content["content"] = None

    # Get data from News table
    n_list = []
    news_params = "order_by=-Created on&filter__field_1156934__link_row_has=" + company_id + "&filter__field_1169565__boolean=true"
    try:
        news = get_all_baserow_data(
            os.getenv("BASEROW_TABLE_COMPANY_NEWS"),
            news_params
        )
        for n in news:
            published_time = datetime.strptime(n['Created on'], "%Y-%m-%dT%H:%M:%S.%fZ")
            formatted_time = published_time.strftime(os.getenv("DATE_FORMAT"))
            unix_time = int(published_time.timestamp())
            news_item = NewsItem(
                headline=n['Headline'],
                url=n['Link'],
                date=formatted_time,
                sort_date= unix_time,
            )
            n_list.append(news_item)

        # sorted_n_list = sorted(n_list, key=lambda d:d['sort_date'], reverse=True)
        content["news"] = n_list

    except Exception as e:
        content["news"] = None

    # Get data from CompanyFundraising table - take advantage of row link here
    try:
        fundraising_params = "filter__field_2209789__link_row_has=" + company_id
        fundraising_data = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY_FUNDRAISING"), fundraising_params)

        fundraising_dict = {}
        fundraising_list = []

        for entry in fundraising_data:
            if entry['Project ID'] is None or len(entry['Project ID']) < 1:
                amount = float(entry["Amount"])
                formatted_amount = '{:,.2f}'.format(amount)

                if entry['Round'] is None:
                    fundraising_round = ""
                else:
                    fundraising_round = entry['Round']['value']

                fundraising_dict = {"funding_type": entry['Type']['value'], "round": fundraising_round, "amount": formatted_amount, "date": entry["Date"], "year": entry["Date"].split('-')[0], "url": entry["Link"]}
                fundraising_list.append(fundraising_dict) 

            # Get Giveth data
            elif entry['Project ID'] is not None and entry['Type']['value'] == "Giveth":          
                giveth_data = get_giveth_data(entry['Project ID'])                
                fundraising_list.append(giveth_data)
            else:
                pass
        
        fundraising_sums = calculate_dict_sums(fundraising_list)
        content["fundraising"] = fundraising_sums
    except Exception as e:
        content["fundraising"] = None

    # Token data
    try:
        token_list = []

        if result['Token'] is None or len(result['Token']) < 1:
            token = None
        else:
            token_id = result['Token']

            if re.search(r'^[^:]+:[^:]+$', token_id):
                network = token_id.split(':')[0]
                token_address = token_id.split(':')[1]
                r = get_coingeckoterminal_data(network, token_address)
                token_data = r['data']['attributes']

                t = Token(
                    symbol=token_data['symbol'].upper(),
                    price_usd=round(float(token_data['price_usd']),5),
                    percent_change=0,
                    token_id=""
                )
                token_list.append(vars(t))
            else:
                r = get_coingecko_data(token_id)

                for token in r:
                    if token['price_change_percentage_24h'] is None:
                        percent_change = 0
                    else:
                        percent_change = round(token['price_change_percentage_24h'], 2)
                        
                    t = Token(
                        symbol=token['symbol'].upper(),
                        price_usd=round(token['current_price'],5),
                        percent_change=percent_change,
                        token_id=token['id']
                    )
                    token_list.append(vars(t))
        content["token"] = token_list
    except Exception as e:
        content["token"] = None

    # Karma GAP milestone data
    try:
        if result['Karma slug'] is None or len(result['Karma slug']) < 1:
            sorted_activity_list = None
        else:
            activity_list = []
            completed_msg = None
            karma_slug = result['Karma slug']
            karma_data = get_karma_gap_data(karma_slug)

            for grant in karma_data['grants']:
                for m in grant['milestones']:
                    due_date = datetime.fromtimestamp(m['data']['endsAt']).strftime(os.getenv("DATE_FORMAT"))
                    description = markdown.markdown(m['data']['description'])
                    if "completed" in m.keys():
                        status = "Completed"
                        if "reason" in m['completed']['data'].keys():
                            completed_msg = markdown.markdown(m['completed']['data']['reason'])
                            if "proofOfWork" in m['completed']['data'].keys():
                                completed_msg += "\n\n" + "<a href=" + "'" + m['completed']['data']['proofOfWork'] + "'" + "target='_blank'>" + m['completed']['data']['proofOfWork'] + "</a>"
                        else:
                            completed_msg = None
                    elif datetime.fromtimestamp(m['data']['endsAt']) > datetime.now():
                        status = "In Progress"
                    elif datetime.fromtimestamp(m['data']['endsAt']) < datetime.now():
                        status = "Overdue"
                    else:
                        status = "In Progress"

                    milestone = Activity(
                        name=m['data']['title'],
                        description=description,
                        status=status,
                        due_date=due_date,
                        due_date_unix=m['data']['endsAt'],
                        completed_msg=completed_msg,
                        type="Milestone"
                    )
                    activity_list.append(vars(milestone))
                
                for u in grant['updates']:
                    description = markdown.markdown(u['data']['text'])
                    if 'data' in u and 'proofOfWork' in u['data']:
                        description += "<a href=" + "'" + u['data']['proofOfWork'] + "'" + "target='_blank'>" + u['data']['proofOfWork'] + "</a>"
                    due_date_string = datetime.strptime(u['createdAt'],"%Y-%m-%dT%H:%M:%S.%fZ")
                    due_date_unix = datetime.timestamp(due_date_string) 

                    update = Activity(
                        name=u['data']['title'],
                        description=description,
                        status=None,
                        due_date=None,
                        due_date_unix=due_date_unix,
                        completed_msg=None,
                        type="Update")
                    activity_list.append(vars(update))
            
            for update in karma_data['updates']:
                completed_msg = ""
                update_status = None
                if 'deliverables' in update['data']:
                    for d in update['data']['deliverables']:
                        completed_msg += markdown.markdown("- [" + d['name'] + "](" + d['proof'] + ")")
                        update_status = "Delivered"
                description = markdown.markdown(update['data']['text'])
                due_date_string = datetime.strptime(update['createdAt'],"%Y-%m-%dT%H:%M:%S.%fZ")
                due_date_unix = datetime.timestamp(due_date_string)          
                update = Activity(
                    name=update['data']['title'],
                    description=description,
                    status=update_status,
                    due_date=None,
                    due_date_unix=due_date_unix,
                    completed_msg=completed_msg,
                    type="Update")
                activity_list.append(vars(update))
                
            sorted_activity_list = sorted(activity_list, key=lambda d: d['due_date_unix'], reverse=True)
        content["activity"] = sorted_activity_list
    except Exception as e:
        content["activity"] = None

    # Get Company Impact table data
    impact = []
    try:
        impact = get_project_metrics_data(baserow_id=company_id)
        content["impact"] = impact
    except Exception as e:
        content["impact"] = None

    return content

# -----------------------------
# Routes
# -----------------------------

@app.get("/", summary="Root endpoint", dependencies=[])
@limiter.limit("20/minute")
def read_root(request: Request):
    return {
        "message": "CARBON Copy API. See the documentation at /docs",
        "docs_url": "/docs"
    }


@app.get(
    "/impact/projects",
    dependencies=[Depends(get_api_key)],
    response_model=List[ImpactProjectSummary],
    summary="List all projects with data integrated to CARBON Copy",
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
@limiter.limit("20/minute")
def get_impact_projects(request: Request):
    with transaction.atomic():
        projects = Project.objects.all().order_by("name")
    return [
        ImpactProjectSummary(
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
    "/impact/projects/{baserow_id}/metrics",
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
@limiter.limit("20/minute")
def get_project_metrics_data(baserow_id: int, request: Request):
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
    "/impact/feed",
    dependencies=[],
    summary="Provides a feed of updates from Karma",
    responses={
        200: {
            "description": "Provides a feed of updates from Karma",
            "content": {
                "application/json": {
                    "example":
                        [
                            {
                                "id": "0x8122f0204829e93deb20e4e0095080db6a7548fa1ef60147631a834eda6b4e73",
                                "title": "October 2025",
                                "project": "CARBON Copy",
                                "created_date": "November 25, 2025",
                                "sort_date": 1764017698,
                                "details": "<p>Spent most of October polishing the new version of CARBON Copy</p>"
                            }
                        ]
                    }
                }
            }
        }
)
@limiter.limit("20/minute")
def get_impact_feed(request: Request):
    file_path = os.path.join(settings.STATIC_ROOT, "impact_feed.json")
    with open(file_path, "r") as _file:
        data = json.load(_file)

    return data

@app.get(
    "/projects",
    dependencies=[Depends(get_api_key)],
    response_model=List[ProjectSummary],
    summary="List all projects in the CARBON Copy database",
    responses={
        200: {
            "description": "List of projects in the CARBON Copy database with basic info",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "name": "Solar Energy Initiative",
                            "description": "Solar Energy Initiative description",
                            "location": "India",
                            "logo": "https://example.com/logo.png",
                            "karma_slug": "karma-slug-example",
                            "sdg": [{"id": 4440353, "value": "Goal 17 - Partnerships for the Goals", "color": "light-cyan"}],
                            "slug": "solar-energy-initiative",
                            "categories": [{"name": "Renewable Energy", "slug": "renewable-energy"}],
                            "categories": [{"name": "Renewable Energy", "slug": "renewable-energy"}],
                            "links": [{"platform": "Website", "url": "https://www.url.com/", "icon": "globe"}],
                            "protocol": ["Ethereum"],
                            "founders": [{"name": "John Doe", "platforms": [{"platform": "twitter-x", "url": "https://x.com/username"}]}],
                            "coverage": [{"headline": "Headline", "url": "https://url.com", "date": "December 08, 2023", "sort_date": 1701981625}]
                        },
                        {
                            "id": 2,
                            "name": "Wind Farm Alpha",
                            "description": "Wind Farm Alpha description",
                            "location": "USA",
                            "logo": "https://example.com/windfarm.png",
                            "karma_slug": "karma-slug-example",
                            "sdg": [{"id": 4440353, "value": "Goal 17 - Partnerships for the Goals", "color": "light-cyan"}],
                            "slug": "wind-farm-alpha",
                            "categories": [{"name": "Renewable Energy", "slug": "renewable-energy"}],
                            "links": [{"platform": "Website", "url": "https://www.url.com/", "icon": "globe"}],
                            "protocol": ["Ethereum"],
                            "founders": [{"name": "John Doe", "platforms": [{"platform": "twitter-x", "url": "https://x.com/username"}]}],
                            "coverage": [{"headline": "Headline", "url": "https://url.com", "date": "December 08, 2023", "sort_date": 1701981625}]
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_projects(request: Request):
    p_list = []
    file_path = os.path.join(settings.STATIC_ROOT, "projects.json")
    with open(file_path, "r") as _file:
        data = json.load(_file)
    
    for p in data:
        project = ProjectSummary(
            id=p["id"],
            slug=p["slug"],
            name=p["name"],
            logo=p["logo"],
            description=p["description"],
            location=p["location"],
            karma_slug=p["karma_slug"],
            sdg=p.get("sdg", []),
            categories=p.get("categories", []),
            links=p.get("links", []),
            protocol=p.get("protocol", []),
            founders=p.get("founders", []),
            coverage=p.get("coverage", []),
            token=p["token"]
        )

        p_list.append(vars(project))
            
    sorted_p_list = sorted(p_list, key=lambda x:x['name'].lower())

    return sorted_p_list

@app.get(
    "/landscape",
    dependencies=[Depends(get_api_key)],
    summary="List all projects in the CARBON Copy database",
    responses={
        200: {
            "description": "List of projects in the CARBON Copy database with basic info",
            "content": {
                "application/json": {
                    "categories": [
                        {
                            
                        }
                    ],
                    "sdg": [
                        {
                            
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_landscape(request: Request):
    p_list = get_projects(request)

    categories_map = {}
    sdg_dict = {}

    for project in p_list:
        # Iterate over each category in the project's categories
        for category in project['categories']:
            cat_name = category['name']  # Use a hashable key (assuming 'name' is unique)
            if cat_name not in categories_map:
                categories_map[cat_name] = {'category': category['name'], 'slug': category['slug'], 'description': category['description'], 'projects': []}
            categories_map[cat_name]['projects'].append(project)
            
        # Iterate over each SDG in the project's SDGs
        for sdg in project['sdg']:
            key = (sdg['sdg'], sdg['sort_id'])
            sdg_dict.setdefault(key, []).append(project)
    
    categories_list = list(categories_map.values())
    sorted_categories_list = sorted(categories_list, key=lambda x: x['category'].lower())

    sdg_list = [{'sdg': key[0], 'projects': landscape_list, 'sort_id': key[1]} for key, landscape_list in sdg_dict.items()]
    sorted_sdg_list = sorted(sdg_list, key=lambda x: x['sort_id'])

    result = {
        "categories": sorted_categories_list,
        "sdg": sorted_sdg_list
    }

    return result

@app.get(
    "/tokens",
    dependencies=[Depends(get_api_key)],
    summary="List all ReFi project tokens in the CARBON Copy database",
    responses={
        200: {
            "description": "List of ReFi project tokens in the CARBON Copy database",
            "content": {
                "application/json": {
                    "count": [
                        {
                            
                        }
                    ],
                    "tokens": [
                        {
                            
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_token_list(request: Request):
    token_list = []
    token_data_list = []
    combined_list = []
    cg_list = ""
    params = "filter__field_2250961__not_empty&filter__field_1248804__not_empty&include=Name,Slug,Token,Logo"
    tokens = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY"), params)
    
    for token in tokens:
        token_id = token['Token']
        
        if re.search(r'^[^:]+:[^:]+$', token_id):
            cgt_token_dict = {'project': token['Name'], 'slug': token['Slug'], 'token_id': token_id, 'logo': token['Logo'], 'url': None}
            token_list.append(cgt_token_dict)

            network = token_id.split(':')[0]
            token_address = token_id.split(':')[1]
            r = get_coingeckoterminal_data(network, token_address)
            cgt_token_data = r['data']['attributes']
            
            t = Token(
                symbol=cgt_token_data['symbol'].upper(),
                price_usd=round(float(cgt_token_data['price_usd']),5),
                percent_change=0,
                token_id=token_id
            )
            cgt_token_dict = vars(t)
            token_data_list.append(cgt_token_dict)
            
        else:
            if re.search(r'^[a-zA-Z0-9,-]+,+[a-zA-Z0-9,-]+$', token_id):
                tokens = token_id.split(",")
                for t in tokens:
                    cg_list += t + ','
                    token_url = os.getenv("COINGECKO_BASE_URL") + t
                    token_dict = {'project': token['Name'], 'slug': token['Slug'], 'token_id': t, 'logo': token['Logo'], 'url': token_url}
                    token_list.append(token_dict)
            else:
                cg_list += token_id + ',' 
                token_url = os.getenv("COINGECKO_BASE_URL") + token_id
                token_dict = {'project': token['Name'], 'slug': token['Slug'], 'token_id': token_id, 'logo': token['Logo'], 'url': token_url}
                token_list.append(token_dict)

    token_data = get_coingecko_data(cg_list)

    # Process data from CoinGecko

    for token in token_data:
        if token['price_change_percentage_24h'] is None:
            percent_change = 0
        else:
            percent_change = round(token['price_change_percentage_24h'], 2)
        t = Token(
            symbol=token['symbol'].upper(),
            price_usd=round(token['current_price'],5),
            percent_change=percent_change,
            token_id=token['id'])
        
        token_data_list.append(vars(t))

    # Combine lists

    for item1 in token_list:
        matching_item = next((item2 for item2 in token_data_list if item2['token_id'] == item1['token_id']), None)
        if matching_item:
            combined_dict = {**item1, **matching_item}  # Merge dictionaries
            combined_list.append(combined_dict)

    sorted_combined_list = sorted(combined_list, key=lambda x:x['project'].lower())
    token_count = len(token_data_list)

    result = {
        "count": token_count,
        "tokens": sorted_combined_list
    }

    return result

@app.get(
    "/news",
    dependencies=[Depends(get_api_key)],
    summary="List all ReFi project news in the CARBON Copy database",
    responses={
        200: {
            "description": "List of ReFi project news in the CARBON Copy database",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "headline":"Lemonade Foundation chooses Etherisc to protect 100 million farmers by 2030",
                            "company":"Etherisc",
                            "url":"https://x.com/etherisc/status/1992242282557264283?s=20",
                            "date":"November 27, 2025",
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_news_list(request: Request):
    news_list = []

    params = "&size=100&order_by=-Created on"
    data = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY_NEWS"), params)

    for item in data:
        if item['Display'] is True:
            published_time = datetime.strptime(item['Created on'], "%Y-%m-%dT%H:%M:%S.%fZ")
            formatted_time = published_time.strftime(os.getenv("DATE_FORMAT"))
            news = NewsItem(
                headline=item['Headline'],
                company=item['Company'][0]['value'],
                url=item['Link'],
                date=formatted_time
            )
            
            news_list.append(vars(news))
        else:
            continue
    
    return news_list

@app.get(
    "/content/newsletter",
    dependencies=[Depends(get_api_key)],
    summary="List all articles from the CARBON Copy newsletter",
    responses={
        200: {
            "description": "List all articles from the CARBON Copy newsletter",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "title":"Gitcoin Grants 24, $AZUSD, and Impact Stablecoin Tracker",
                            "url":"https://paragraph.com/@carboncopy/gitcoin-grants-24-azusd-and-impact-stablecoin-tracker",
                            "mainImage":"https://storage.googleapis.com/papyrus_images/90d4cd3d0dec49135b7818a0314d0527c77b179c419c2299a12283d549daa0c5.jpg",
                            "date":"October 27, 2025"
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_cc_newsletter(request: Request):
    newsletter_list = []

    r = requests.get("https://paragraph.xyz/api/blogs/rss/@carboncopy")

    f = feedparser.parse(r.text)

    for article in f.entries[0:3]:
        mainImage = ""
        date = parse_datetime(article.published)
        formatted_date = date.strftime(os.getenv("DATE_FORMAT"))
        for link in article.links:
            if link.type == "image/jpg" or link.type == "image/jpeg":
                mainImage = link.href

        a = Article(
            title=article.title,
            url=article.link,
            mainImage=mainImage,
            publication="Paragraph",
            date=formatted_date
            )
        
        newsletter_list.append(vars(a))

    return newsletter_list

@app.get(
    "/categories/{slug}",
    dependencies=[Depends(get_api_key)],
    summary="List all categories in the CARBON Copy database",
    response_model=CategoryResponse,
    responses={
        200: {
            "description": "List of categories in the CARBON Copy database",
            "content": {
                "application/json": {
                    "example": [
                        {
                            
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def category_projects(slug, request: Request):
    p_list = []
    comp_list = []
    category_news_list = []
    category_fundraising_list = []
    token_list = []
    filter_string = ""
    tokens = ""
    data = get_landscape()
    categories = data["categories"]

    try:
        result = next((cat for cat in categories if cat["slug"] == slug), None)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving category information")
    
    if result is None:
        raise HTTPException(status_code=404, detail="Category not found")
    
    metadata = {"name": result['category'], "slug": result['slug'], "description": result['description'], "count": len(result["projects"])}    

    for p in result["projects"]:
        comp_list.append(p['name'])
        project = ProjectSummary(
            id=p["id"],
            name=p['name'],
            slug=p["slug"],
            description=p['description'],
            logo=p['logo'],
            location=p['location'],
            karma_slug=p["karma_slug"],
            categories=p["categories"],
            sdg=p["sdg"],
            links=p["links"],
            founders=p["founders"],
            coverage=p["coverage"],
            protocol=p["protocol"],
            token=p["token"],
        )
        p_list.append(vars(project))
        
        if project.token:
            tokens += project.token + ","

    sorted_p_list = sorted(p_list, key=lambda x:x['name'].lower())

    if tokens != "":
        try:
            token_data = get_coingecko_data(tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error getting data from Coingecko.")

        for token in token_data:
            if token['price_change_percentage_24h'] is None:
                percent_change = 0
            else:
                percent_change = round(token['price_change_percentage_24h'], 2)
            url = os.getenv("COINGECKO_BASE_URL") + token['id']
            token = Token(
                image=token['image'],
                symbol=token['symbol'].upper(),
                price_usd=round(token['current_price'],5),
                percent_change=percent_change,
                url=url,
                token_id=token["id"],
            )
            token_list.append(vars(token))
    else:
        pass

    # Get a list of news items related to projects in the category
    try:
        news_data = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY_NEWS"),"size=50&order_by=-Created on", single_page=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading news items")
    news_list = [
        news_item for news_item in news_data
        if any(project['value'] in comp_list for project in news_item["Company"])
    ]

    # Create a list of news item dicts
    for item in news_list[0:60]:
        published_time = datetime.strptime(item['Created on'], "%Y-%m-%dT%H:%M:%S.%fZ")
        formatted_time = published_time.strftime(os.getenv("DATE_FORMAT"))
        category_news = NewsItem(
            headline=item['Headline'],
            date=formatted_time,
            url=item['Link'],
            company=item['Company'][0]["value"]
        )
        
        category_news_list.append(vars(category_news))

    # Get fundraising data
    for p in comp_list:
        filter_string += "filter__field_2209789__link_row_contains=" + p + "&"

    fundraising_params = "filter_type=OR&" + filter_string
    try:
        fundraising_data = get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY_FUNDRAISING"), fundraising_params)
    except Exception as e:
        raise HTTPException(status_code=500, details="Could not retrieve fundraising data")
    fundraising_list = [
        fundraising_item for fundraising_item in fundraising_data
        if any(project['value'] in comp_list for project in fundraising_item["Company"])
    ]

    for f in fundraising_list:
        if f['Project ID'] is None or len(f['Project ID']) < 1:
            funding_type = f['Type']['value']
            fundraising_dict = {
                "funding_type": funding_type,
                "amount": f['Amount'],
                "round": None if f['Round'] is None else f['Round']['value'],
                "date": f["Date"],
                "year": f["Date"].split('-')[0],
                "url": f['Link']
            }
            category_fundraising_list.append(fundraising_dict)

        # Get Giveth data
        if f['Project ID'] is not None and f['Type']['value'] == "Giveth":
            try:          
                giveth_data = get_giveth_data(f['Project ID'])   
            except Exception as e:
                raise HTTPException(status_code=500, detail="Could not retrieve Giveth data")             
            category_fundraising_list.append(giveth_data)
        else:
            pass

    fundraising_sums = calculate_dict_sums(category_fundraising_list)

    category = vars(CategoryResponse(
        metadata=metadata,
        projects=sorted_p_list,
        tokens=token_list,
        news=category_news_list,
        fundraising=fundraising_sums))

    return category

@app.get(
    "/projects/{project_slug}",
    dependencies=[Depends(get_api_key)],
    response_model=ProjectSummary,
    summary="Details for a project in the CARBON Copy database",
    responses={
        200: {
            "description": "Details for a project in the CARBON Copy database with basic info",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "name": "Solar Energy Initiative",
                            "description": "Solar Energy Initiative description",
                            "location": "India",
                            "logo": "https://example.com/logo.png",
                            "karma_slug": "karma-slug-example",
                            "sdg": [{"id": 4440353, "value": "Goal 17 - Partnerships for the Goals", "color": "light-cyan"}],
                            "slug": "solar-energy-initiative",
                            "categories": [{"name": "Renewable Energy", "slug": "renewable-energy"}],
                            "categories": [{"name": "Renewable Energy", "slug": "renewable-energy"}],
                            "links": [{"platform": "Website", "url": "https://www.url.com/", "icon": "globe"}],
                            "protocol": ["Ethereum"],
                            "founders": [{"name": "John Doe", "platforms": [{"platform": "twitter-x", "url": "https://x.com/username"}]}],
                            "coverage": [{"headline": "Headline", "url": "https://url.com", "date": "December 08, 2023", "sort_date": 1701981625}]
                        }
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_project_details(project_slug: str, request: Request):
    p_list = []
    file_path = os.path.join(settings.STATIC_ROOT, "projects.json")
    with open(file_path, "r") as _file:
        data = json.load(_file)

    result = next((item for item in data if item["slug"] == project_slug), None)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")
    return result

@app.get(
    "/projects/{project_slug}/content",
    dependencies=[Depends(get_api_key)],
    # response_model=ProjectSummary,
    summary="Details for a project in the CARBON Copy database",
    responses={
        200: {
            "description": "Details for a project in the CARBON Copy database with basic info",
            "content": {
                "application/json": {
                    "example": [
                        
                    ]
                }
            }
        }
    }
)
@limiter.limit("20/minute")
def get_dynamic_project_details(project_slug: str, request: Request):
    content = dynamic_project_content(slug=project_slug)
    return content

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
@limiter.limit("20/minute")
def get_aggregate_metric_types(request: Request):
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
@limiter.limit("20/minute")
def aggregate_metric_type_endpoint(slug: str, request: Request) -> AggregateMetricTypeResponse:
    with transaction.atomic():
        try:
            obj = AggregateMetricType.objects.get(slug=slug)
        except AggregateMetricType.DoesNotExist:
            raise HTTPException(
                status_code=404,
                detail="Invalid aggregate metric type"
            )
        aggs = AggregateMetric.objects.filter(type=obj)

        result = _build_metric_group(
            aggs,
            chart_flag_field="chart",
            include_pie=True,
            metric_type=obj,
        )

        return AggregateMetricTypeResponse(
            type_name=obj.name,
            description=obj.description,
            projects_count=Project.objects.filter(
                metrics__aggregate_metric__in=aggs
            ).distinct().count(),
            **result,
        )

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
@limiter.limit("20/minute")
def get_sdg_list(request: Request):
    sdgs = SDG.objects.all()
    sdg_out = []

    for sdg in sdgs:
        agg_qs = (
            AggregateMetric.objects
            .filter(sdg=sdg)
            .select_related("type")
        )

        grouped_metrics = {}

        for agg in agg_qs:
            agg_item, _ = _agg_metric_calc(agg)

            metric_type = agg.type
            type_key = metric_type.slug if metric_type else "uncategorized"

            if type_key not in grouped_metrics:
                grouped_metrics[type_key] = {
                    "type": {
                        "name": metric_type.name if metric_type else "Uncategorized",
                        "slug": metric_type.slug if metric_type else "uncategorized",
                    },
                    "metrics": [],
                }

            grouped_metrics[type_key]["metrics"].append(agg_item)

        sdg_out.append(
            SDGList(
                name=sdg.name,
                description=sdg.description,
                slug=sdg.slug,
                metric_groups=list(grouped_metrics.values()),
            )
        )

    return sdg_out

@app.get(
    "/sdg/{slug}",
    dependencies=[],
    response_model=SDGDetailResponse,
    summary="Get aggregate metric data by SDG",
    description="Returns the aggregate metrics for the given SDG slug. The SDG slug must exist in the SDG table."
)
@limiter.limit("20/minute")
def get_sdg_detail(slug: str, request: Request) -> SDGDetailResponse:
    with transaction.atomic():
        sdg = SDG.objects.get(slug=slug)

        agg_qs = (
            AggregateMetric.objects
            .filter(sdg=sdg)
            .select_related("type")
        )

    grouped = defaultdict(list)
    for agg in agg_qs:
        grouped[agg.type].append(agg)

    groups_out = []

    for metric_type, aggs in grouped.items():
        result = _build_metric_group(
            aggs,
            chart_flag_field="sdg_chart",
        )

        groups_out.append(
            SDGMetricGroup(
                type={
                    "name": metric_type.name if metric_type else "Uncategorized",
                    "description": metric_type.description if metric_type else "Aggregate metrics don't have a type",
                    "slug": metric_type.slug if metric_type else "uncategorized",
                },
                **result,
            )
        )

    return SDGDetailResponse(
        name=sdg.name,
        description=sdg.description,
        slug=sdg.slug,
        groups=groups_out,
    )

@app.get(
    "/overview",
    dependencies=[Depends(get_api_key)],
    response_model=OverviewResponse,
    summary="Overview of Funding Metrics"
)
@limiter.limit("20/minute")
def get_overview(request: Request):
    return get_overview_data()

@app.get(
    "/venture-funding",
    dependencies=[Depends(get_api_key)],
    response_model=VentureFundingResponse,
    summary="Venture Funding Overview",
    description="Returns total venture funding, deals, charts, project breakdown, and current year deals"
)
@limiter.limit("20/minute")
def venture_funding_endpoint(request: Request):
    return get_venture_funding_data()

@app.get("/link-preview", dependencies=[Depends(get_api_key)], include_in_schema=False)
@limiter.limit("20/minute")
async def link_preview(url: str, request: Request):
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

# @app.post("/altcha", include_in_schema=False)
# @limiter.limit("10/minute")
# async def altcha_challenge(request: Request):
#     options = ChallengeOptions(
#         expires=datetime.now() + timedelta(hours=1),
#         max_number=100000, # The maximum random number
#         hmac_key=os.getenv("ALTCHA_HMAC_KEY"),
#     )
#     return create_challenge(options)

# @app.get("/register", response_class=HTMLResponse, include_in_schema=False)
# @limiter.limit("10/minute")
# async def register_form(request: Request):
#     return templates.TemplateResponse("register.html", {"request": request})

# @app.post("/register", response_class=HTMLResponse, include_in_schema=False)
# @limiter.limit("10/minute")
# async def register_api_key(request: Request, name: str = Form(...), altcha: str = Form(...)):
#     # Verify ALTCHA
#     if not altcha.verify(altcha):
#         return templates.TemplateResponse("register.html", {"request": request, "error": "CAPTCHA verification failed. Try again."})
    
#     try:
#         with transaction.atomic():
#             api_key = APIKey(name=name)
#             api_key.save()
#         return templates.TemplateResponse("success.html", {"request": request, "key": api_key.key})
#     except Exception as e:
#         return templates.TemplateResponse("register.html", {"request": request, "error": "Registration failed. Try again."})
