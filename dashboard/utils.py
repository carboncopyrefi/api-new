
import requests, os
from django import forms
from .models import ProjectMetric

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

def get_all_baserow_data(table_id: str, params: str) -> list[dict]:
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