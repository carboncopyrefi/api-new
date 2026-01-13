import json
import os
from datetime import datetime
from django.conf import settings
from dashboard import utils, schemas

DATE_FORMAT = os.getenv("DATE_FORMAT")

def generate_projects_json() -> str:

    founders = utils.get_all_baserow_data(
        os.getenv("BASEROW_TABLE_COMPANY_FOUNDER"),
        "Links__join=URL&filter__field_1139228__not_empty"
    )

    # news = utils.get_all_baserow_data(
    #     os.getenv("BASEROW_TABLE_COMPANY_NEWS"),
    #     "order_by=-Created on"
    # )

    projects = utils.get_all_baserow_data(
        os.getenv("BASEROW_TABLE_COMPANY"),
        (
            f"filter__field_1248804__not_empty"
            f"&Links__join=URL"
            f"&Category__join=Name,Slug"
            f"&Coverage__join=Headline,Link,Publication,Publish Date"
        )
    )

    p_list = []

    for result in projects:
        company_name = result["Name"]

        # # Get data from News table
        # n_dict = {}
        # n_list = []

        # for n in news:
        #     if any(c['value'] == company_name for c in n['Company'] ):
        #         published_time = datetime.strptime(n['Created on'], "%Y-%m-%dT%H:%M:%S.%fZ")
        #         formatted_time = published_time.strftime(DATE_FORMAT)
        #         unix_time = int(published_time.timestamp())
        #         n_dict = {"headline": n['Headline'], "url": n['Link'], "date": formatted_time, "sort_date": unix_time}
        #         n_list.append(n_dict)
        #     else:
        #         pass

        # sorted_n_list = sorted(n_list, key=lambda d:d['sort_date'], reverse=True)

        # Links
        links = [
            {
                "platform": l["value"],
                "url": l["URL"],
                "icon": utils.contact_icon(l["value"].lower()),
            }
            for l in result.get("Links", [])
        ]

        # Founders
        founders_out = []
        for f in founders:
            if any(c["value"] == company_name for c in f["Company"]):
                founders_out.append(
                    {
                        "name": f["Name"],
                        "platforms": [
                            {
                                "platform": utils.contact_icon(l["value"].lower()),
                                "url": l["URL"],
                            }
                            for l in f.get("Links", [])
                        ],
                    }
                )

        # Categories
        categories = [
            {"name": c["Name"], "slug": c["Slug"]}
            for c in result.get("Category", [])
        ]

        # Coverage
        coverage = []
        for c in result.get("Coverage", []):
            published = datetime.strptime(c["Publish Date"], "%Y-%m-%d")
            coverage.append(
                {
                    "headline": c["Headline"],
                    "publication": c["Publication"]["value"],
                    "url": c["Link"],
                    "date": published.strftime("%Y-%m-%d"),
                    "sort_date": int(published.timestamp()),
                }
            )

        coverage.sort(key=lambda x: x["sort_date"], reverse=True)

        # Protocol
        protocol_list = []
        for p in result.get("Protocol", []):
            protocol_list.append(p["value"])

        project = schemas.ProjectSummary(
            id = result["id"],
            slug = result["Slug"],
            name = company_name,
            logo = result["Logo"],
            description = result["One-sentence Description"],
            links = links,
            categories = categories,
            founders = founders_out,
            # news = sorted_n_list,
            coverage = coverage,
            location = result["Location"],
            protocol = protocol_list,
            karma_slug = result["Karma slug"],
            sdg = result.get("SDG"),
        )

        p_list.append(vars(project))

    # Write file to static directory
    output_dir = os.path.join(settings.STATIC_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "projects.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(p_list, f, indent=2)

    return output_path
