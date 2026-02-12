from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime

# -----------------------------
# Pydantic Models
# -----------------------------
class ImpactProjectSummary(BaseModel):
    """Summary information for a project."""
    name: str = Field(..., example="Solar Energy Initiative")
    logo_url: Optional[str] = Field(None, example="https://example.com/logo.png")
    metrics: List[str] = Field(..., example=["Installed Capacity", "CO2 Savings"])
    slug: Optional[str] = Field(None, example="solar-energy-initiative")

class ProjectSummary(BaseModel):
    id: int = Field(..., description="Unique identifier for the project")
    slug: str = Field(..., description="URL-friendly slug for the project")
    name: str = Field(..., description="Name of the project")
    logo: str = Field(..., description="URL to the project's logo image")
    description: str = Field(..., description="One-sentence description of the project")
    location: Optional[str] = Field(None, description="Location of the project")
    karma_slug: Optional[str] = Field(None, description="Karma slug for the project")
    sdg: Optional[List] = Field(None, description="List of SDGs associated with the project")
    categories: Optional[List] = Field(None, description="List of categories for the project")
    links: Optional[List] = Field(None, description="List of links associated with the project")
    founders: Optional[List] = Field(None, description="List of founders for the project")
    coverage: Optional[List] = Field(None, description="List of coverage articles related to the project")
    protocol: Optional[List] = Field(None, description="Protocol(s) used by the project")
    token: Optional[str] = Field(None, example="ether")

class Article(BaseModel):
    title: str = Field(..., example="New Milestone Achieved")
    url: str = Field(..., example="https://url.com/articles/new-milestone-achieved")
    mainImage: Optional[str] = Field(None, example="https://example.com/image.png")
    publication: Optional[str] = Field(None, example="CoinDesk")
    date: str = Field(..., example="July 20, 2025")

class NewsItem(BaseModel):
    headline: str = Field(..., example="News Headline")
    company: Optional[str] = Field(None, example="Company Name")
    url: str = Field(..., example="https://url.com/articles/new-milestone-achieved")
    date: str = Field(..., example="July 20, 2025")
    sort_date: Optional[int] = Field(None, example=349304832)

class Token(BaseModel):
    symbol: str = Field(..., example="ETH")
    price_usd: float = Field(..., example=2500.0)
    percent_change: Optional[float] = Field(None, example=2.5)
    token_id: str = Field(..., example="ethereum")
    image: Optional[str] = Field(None, exmaple="https://image.com/image.png")
    url: Optional[str] = Field(None, example="https://coingecko.com/token")

class Activity(BaseModel):
    name: str = Field(..., example="New Milestone Achieved")
    description: str = Field(..., example="Description of New Milestone Achieved")
    status: Optional[str] = Field(None, example="In Progress")
    due_date: Optional[str] = Field(None, example="January 31, 2026")
    due_date_unix: int = Field(..., example=349304832)
    completed_msg: Optional[str] = Field(None, example="We completed the new mulestone.")
    type: str = Field(..., example="Milestone")

class Update(BaseModel):
    id: str = Field(..., example="0x5189ee3c3469e6115a0252c5441c678e926011fb06cd4674527aaf632b55d692")
    title: str = Field(..., example="Update #1")
    project: str = Field(..., example="Project Name")
    created_date: Optional[str] = Field(None, example="July 14, 2026")
    sort_date: Optional[int] = Field(None, example="2025-04-30T16:00:00.000Z")
    details: Optional[str] = Field(None, example="<p>Update details")

class CategoryResponse(BaseModel):
    metadata: dict = Field(..., example='{"count":8,"description":"Standalone cryptocurrencies and stablecoins aimed primarily at making a positive ecological and/or social impact.","name":"Impact Currency","slug":"impact-currency"}')
    projects: List[ProjectSummary] = Field(..., example='[{"location":"Peru","logo":"https://pbs.twimg.com/profile_images/1507695433199632399/XzLPrKxu_400x400.jpg","name":"Alinticoin","short_description":"Alinticoin finances the future of electricity from the photosynthesis of plants.","slug":"alinticoin"]')
    tokens: List[Token] = Field(..., example="")
    news: List[NewsItem] = Field(..., example="")
    fundraising: List[dict] = Field(..., example='[{"amount":"46,697.09","details":[{"amount":"678.84","date":"2025-02-05","funding_type":"Gitcoin Grants","round":"Glo Dollar x Optimism Builders","url":"","year":"2025"}]')

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
    chart: Optional[list] = Field(None, description="Time-series chart data with cumulative values", example=[{"month": "2025-01", "Installed Capacity": 25.4}])

class AggregateMetricTypeList(BaseModel):
    name: str = Field(..., example="Total Installed Capacity")
    description: Optional[str] = Field(None, example="Sum of installed capacity across all projects")   
    slug: str = Field(..., example="total-installed-capacity")
    pie_chart: str = Field(..., example="Project", description="Pie chart grouping for this metric type")

class AggregateMetricTypeOut(BaseModel):
    name: str = Field(..., example="Total Installed Capacity")
    description: Optional[str] = Field(None, example="Sum of installed capacity across all projects")   
    slug: str = Field(..., example="total-installed-capacity")

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

class AggregateMetricGroup(BaseModel):
    type: AggregateMetricTypeOut
    metrics: List[AggregateMetricItem]

class SDGList(BaseModel):
    name: str = Field(..., example="Goal #1 - No Poverty")
    description: Optional[str] = Field(None, example="SDG description")   
    slug: str = Field(..., example="1-no-poverty")
    metric_groups: List[AggregateMetricGroup]

class SDGMetricGroup(BaseModel):
    type: AggregateMetricTypeOut
    metrics: List[AggregateMetricItem]
    table: AggregateMetricTypeTable
    charts: Optional[List[dict]] | None = None

class SDGDetailResponse(BaseModel):
    name: str
    description: str | None
    slug: str
    groups: List[SDGMetricGroup]

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
    table: Optional[AggregateMetricTypeTable]
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

class FundraisingItem(BaseModel):
    type: str = Field(..., example="Grant")
    type_id: int = Field(..., example=1)
    amount: float = Field(..., example=5000.0)
    date: Optional[str] = Field(None, example="2025-07-15")
    project: str = Field(..., example="Project Name")
    reference_url: Optional[str] = Field(None, example="https://example.com/grant-details")

class FundingOverviewResponse(BaseModel):
    venture_funding: VentureFundingResponse
    pgf_funding: VentureFundingResponse