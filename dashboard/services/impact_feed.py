import markdown, requests, os, json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dashboard import utils, schemas
from django.conf import settings

baserow_table_company = os.getenv("BASEROW_TABLE_COMPANY")
date_format = os.getenv("DATE_FORMAT")

def generate_impact_json() -> str:
    file_path = os.path.join(settings.STATIC_ROOT, "projects.json")
    with open(file_path, "r") as _file:
        data = json.load(_file)

    result = [project for project in data if project.get("karma_slug")]

    current_timestamp = datetime.now()
    three_months_ago = current_timestamp - timedelta(days=90)

    def fetch_updates(project):
        local_updates = []
        slug = project['karma_slug']
        api = f"https://gapapi.karmahq.xyz/v2/projects/{slug}"
        
        try:
            response = requests.get(api, timeout=20)
            response.raise_for_status()
        except Exception as e:
            print(f"Request failed for {slug}: {e}")
            return []

        updates = response.json() or []

        for update in updates['updates']:
            created_date_str = update.get('createdAt')
            if not created_date_str:
                continue

            try:
                date_created = datetime.strptime(created_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                continue

            if not (three_months_ago <= date_created <= current_timestamp):
                continue
            id = update['uid']
            title = update.get('title', 'Untitled')
            date = date_created.strftime(date_format)
            details = markdown.markdown(update.get('text', '') + '<p class="fw-bold">Deliverables</p>')

            for deliverable in update.get('deliverables', []):
                details += markdown.markdown(f"- [{deliverable['name']}]({deliverable['proof']})")

            item = schemas.Update(
                id=id,
                title=title,
                project=project['name'],
                created_date=date,
                sort_date=int(date_created.timestamp()),
                details=details,
            )
            
            local_updates.append(vars(item))
        print(local_updates)
        return local_updates

    # Run in parallel
    updates_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_updates, project) for project in result]
        for future in as_completed(futures):
            try:
                updates = future.result()
                if updates:
                    updates_list.extend(updates)
            except Exception as e:
                print(f"Error in future: {e}")

    # Sort the final list
    sorted_updates_list = sorted(updates_list, key=lambda d: d['sort_date'], reverse=True)

    output_dir = os.path.join(settings.STATIC_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "impact_feed.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_updates_list, f, indent=2)

    return output_path