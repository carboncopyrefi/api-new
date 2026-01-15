
import requests, os
from django import forms
from .models import ProjectMetric
from datetime import datetime
from collections import defaultdict

BASEROW_API = "https://api.baserow.io/api/database/rows/table/"
BASEROW_TOKEN = os.getenv("BASEROW_API_KEY")


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label="CSV file")
    project_metric = forms.ModelChoiceField(
        queryset=ProjectMetric.objects.all(),
        required=True,
        label="Metric"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Label: "Metric name (Project name)"
        self.fields["project_metric"].label_from_instance = (
            lambda obj: f"{obj.name} ({', '.join(p.name for p in obj.projects.all())})"
        )

def get_nested_value(data, key_path):
    keys = key_path.split(".")  # Split the key path string by dots
    for key in keys:
        data = data.get(key)  # Access the next level in the nested dict
        if data is None:  # If the key doesn't exist, return None
            return None
    return data


def get_baserow_project_data(baserow_id):
    url = "https://api.baserow.io/api/database/rows/table/171320/" + str(baserow_id) + "/?user_field_names=true&include=Slug,Name,Logo "
    headers = {
        "Authorization": f"Token {BASEROW_TOKEN}",
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch Baserow impact metric data with status {response.status_code}. {response.text}"
        )

def get_all_baserow_data(table_id: str, params: str, single_page: bool = False) -> list[dict]:
    """Fetch all rows from a Baserow table with pagination support."""
    url = f"{BASEROW_API}{table_id}/?user_field_names=true&{params}"
    headers = {
        "Authorization": f"Token {BASEROW_TOKEN}",
        "Content-Type": "application/json",
    }

    all_results = []
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch Baserow data with status {response.status_code}. {response.text}"
            )
        data = response.json()

        if single_page:
            return data["results"]
        
        all_results.extend(data["results"])
        url = data.get("next")  # None when finished

    return all_results

def contact_icon(contact):
    icons = {
        "website": "globe",
        "x": "twitter-x",
        "facebook": "facebook",
        "linkedin": "linkedin",
        "medium": "medium",
        "instagram": "instagram",
        "tiktok": "tiktok",
        "discord": "discord",
        "github": "github",
        "whitepaper": "file-text-fill",
        "blog": "pencil-square",
        "podcast": "broadcast-pin",
        "telegram": "telegram",
        "youtube": "youtube",
        "dao": "bounding-box-circles",
    }
    
    if contact in icons.keys():
        icon = icons[contact]
    
    return icon

def parse_datetime(datetime_str):
    formats = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%B %d, %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Time data '{datetime_str}' does not match any format")

def get_giveth_data(slug):
    query = f"""
        query {{
            projectBySlug(slug:"{ slug }") {{
                id
                totalDonations
                status {{
                    name
                }}
                adminUser {{
                    walletAddress
                }}
            }}
        }}
        """
    api = "https://mainnet.serve.giveth.io/graphql"
    response = requests.post(api, json={'query': query})

    if response.status_code == 200:
        result = response.json()
        total_donations = float(result['data']['projectBySlug']['totalDonations'])
        formatted_total_donations = '{:,.2f}'.format(total_donations)
        formatted_response = {
            "round": "Donations & Matching",
            "amount": formatted_total_donations,
            "funding_type": "Giveth",
            "url": "https://giveth.io/project/" + slug,
            "date": None,
            "year": None
        }
        return formatted_response
    else:
        raise Exception(f"Query failed to run with a {response.status_code}. {response.text}")

def calculate_dict_sums(data):
    amounts_by_type = defaultdict(lambda: {'total_amount': 0, 'details': []})

    for entry in data:
        funding_type = entry["funding_type"]
        amount = float(entry["amount"].replace(',', ''))
        amounts_by_type[funding_type]['total_amount'] += amount
        amounts_by_type[funding_type]['details'].append(entry)

    for funding_type in amounts_by_type:
        amounts_by_type[funding_type]['details'].sort(key=lambda x: (x['date'] is None, x['date']), reverse=True)

    grouped_data = [{"funding_type": funding_type, "amount": '{:,.2f}'.format(info['total_amount']), "details": info['details']}
                for funding_type, info in amounts_by_type.items()]

    return grouped_data

def get_coingecko_data(token_ids):
    header = {
        'x_cg_demo_api_key': os.getenv("COINGECKO_KEY")
    }
    api = "https://api.coingecko.com/api/v3/coins/markets?ids=" + token_ids + "&vs_currency=usd"
    response = requests.get(api, header)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch Coingecko data with status {response.status_code}. {response.text}")
    
def get_coingeckoterminal_data(network, token_id):

    api = "https://api.geckoterminal.com/api/v2/networks/" + network + "/tokens/" + token_id
    response = requests.get(api)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch Coingecko Terminal data with status {response.status_code}. {response.text}")

def get_karma_gap_data(karma_slug):

    api = "https://gapapi.karmahq.xyz/projects/" + karma_slug
    response = requests.get(api)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch Karma GAP data with status {response.status_code}. {response.text}")