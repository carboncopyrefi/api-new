import json
import os
from datetime import datetime
from django.conf import settings
from dashboard import utils, schemas

DATE_FORMAT = os.getenv("DATE_FORMAT")

def generate_fundraising_json() -> str:
    fundraising_list = []
    venture_funding_list = []
    pgf_list = []
    page_size = 200
    params = (
        f"size={page_size}"
        f"&order_by=-Date"
        f"&filter__field_2209789__not_empty"
    )
    venture_type_ids = [1686865, 1688192]
    pgf_type_ids = [1686863, 1706978, 1686864, 1707719, 1715852, 1793489, 2032349, 2371111, 3020666]

    try:
        records = utils.get_all_baserow_data(os.getenv("BASEROW_TABLE_COMPANY_FUNDRAISING"), params)
    except Exception as e:
        print(f"Error fetching fundraising data: {e}")
        return ""

    for r in records:
        if r["Type"]["id"] in venture_type_ids:

            venture_item = schemas.FundraisingItem(
                type=r["Type"]["value"],
                type_id=r["Type"]["id"],
                amount=r["Amount"],
                date=r["Date"],
                project=r["Company"][0]["value"] if r["Company"][0]['value'] else None,
                reference_url=r["Link"],

            )
            venture_funding_list.append(vars(venture_item))

        elif r["Type"]["id"] in pgf_type_ids:
            date = datetime.strptime(r["Date"], "%Y-%m-%d") if r["Date"] else None
            if r['Project ID'] is not None and r['Type']['value'] == "Giveth": 
                try:         
                    giveth_data = utils.get_giveth_data(r['Project ID'])
                except Exception as e:
                    print(f"Error fetching Giveth data for project {r['Project ID']}: {e}")
                    continue                

            pgf_item = schemas.FundraisingItem(
                type=r["Type"]["value"],
                type_id=r["Type"]["id"],
                amount=float(giveth_data['amount'].replace(",", "")) if r['Type']['value'] == "Giveth" else r["Amount"],
                date=r["Date"] if r["Date"] else None,
                project=r["Company"][0]["value"] if r["Company"][0]['value'] else None,
                reference_url=giveth_data["url"] if r['Type']['value'] == "Giveth" else r["Link"],
            )

            pgf_list.append(vars(pgf_item))
        else:
            continue
    
    
    fundraising_list = {
        "venture_funding": venture_funding_list,
        "pgf": pgf_list

    }

    # Write file to static directory
    output_dir = os.path.join(settings.STATIC_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "fundraising.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fundraising_list, f, indent=2)

    return output_path
