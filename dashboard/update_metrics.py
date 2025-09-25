import requests, base64, binascii, json, os

from dune_client.client import DuneClient

from .regen_pb2 import QueryBatchesResponse, QuerySupplyResponse
from .utils import get_nested_value

def get_metric_by_db_id(metrics: list, db_id: int):
    """
    Finds the metric dict from a list of metrics that matches the given db_id.
    Raises ValueError if not found.
    """
    metric = next((m for m in metrics if m["db_id"] == db_id), None)
    if not metric:
        raise ValueError(f"No metric found with db_id={db_id}")
    return metric

def refresh_dune(impact: dict, db_id: int) -> float:
    # Dune initialization
    dune = DuneClient(os.getenv("DUNE_KEY"))

    # Find the metric with matching db_id
    metric = get_metric_by_db_id(impact["metrics"], db_id)

    # Fetch result from Dune
    query = dune.get_latest_result(
        metric["query"],
        max_age_hours=int(metric["max_age"])
    )

    # Extract and round value
    value = round(
        float(query.result.rows[int(metric["result_index"])][metric["result_key"]]),
        2
    )

    # Apply denominator if present
    if metric.get("denominator") is not None:
        value = value / int(metric["denominator"])

    return value


def refresh_client(impact: dict, db_id: int):
    # Find the metric with matching db_id
    metric = get_metric_by_db_id(impact["metrics"], db_id)

    if impact["method"] == "POST":
        post_body = json.loads(json.dumps(impact["body"]))
        response = requests.post(impact["api"], json=post_body)
        metric_data = response.json()[impact["result_key"]][impact["result_index"]]

        value = metric_data[metric["result_key"]]
        formatted_value = (
            round(float(value) / impact["global_denominator"], 2)
            if impact["global_operator"] == "divide"
            else round(float(value), 2)
        )

        if metric["operator"] == "multiply":
            formatted_value = round(formatted_value * metric["denominator"], 2)
        elif metric["operator"] == "divide":
            formatted_value = round(formatted_value / metric["denominator"], 2)

        return formatted_value

    elif impact["method"] == "GET":
        if impact["result_key"] is not None:
            response = requests.get(impact["api"])
            value_path = impact["result_key"] + "." + metric["result_key"]
            value = round(float(get_nested_value(response.json(), value_path)), 2)
            if metric["denominator"] is not None:
                value = value / int(metric["denominator"])
            return value
        else:
            list_value = 0
            api = impact["api"] + metric["query"]
            response = requests.get(api)
            value = response.json()

            if isinstance(value, int):
                if metric["denominator"] is not None:
                    value = float(value) / int(metric["denominator"])
            elif isinstance(value, dict):
                if "list_name" in metric:
                    for i in value[metric["list_name"]]:
                        list_value += float(i[metric["result_key"]])
                    value = list_value
                else:
                    value = float(value[metric["result_key"]])
                if metric["denominator"] is not None:
                    value = value / int(metric["denominator"])

            return round(value, 2)


def refresh_subgraph(impact: dict, db_id: int):
    metric = get_metric_by_db_id(impact["metrics"], db_id)
    cumulative_value = 0

    for q in metric["query"]:
        response = requests.post(
            impact["api"].replace("{api_key}", os.getenv("SUBGRAPH_KEY")) + q,
            json={"query": metric["graphql"]},
        )
        if response.status_code == 200:
            result = response.json()["data"][impact["result_key"]]
            for r in result:
                if r["key"] == metric["result_key"]:
                    cumulative_value += float(r["value"])

    return cumulative_value

def refresh_vebetter(impact: dict, db_id: int):
    metric = get_metric_by_db_id(impact["metrics"], db_id)
    vebetter_api = "https://graph.vet/subgraphs/name/vebetter/dao"

    response = requests.post(
            vebetter_api,
            json={"query": impact["graphql"], "variables": impact["variables"]},
        )
    
    if response.status_code == 200:
        result = response.json()['data']['statsAppSustainabilities'][0]
        value = 0

        if metric['result_key'] in result:
            if metric['result_index'] is not None:
                value = float(result[metric['result_key']][metric['result_index']])
            else:
                value = float(result[metric['result_key']])
        else:
            return None
        
        if metric['operator'] == "divide":
            value = value / metric['denominator']

        if metric['operator'] == "multiply":
            value = value * metric['denominator']

    return value

def refresh_graphql(impact: dict, db_id: int):
    metric = get_metric_by_db_id(impact["metrics"], db_id)
    result_list = []

    if impact["query"] and len(impact["query"]) > 0:
        for q in impact["query"]:
            gql_query = impact["graphql"].replace("{query}", f'"{q}"')
            response = requests.post(impact["api"], json={"query": gql_query})

            if response.status_code == 200:
                if impact["result_index"] is not None:
                    result = response.json()["data"][impact["result_key"]][
                        impact["result_index"]
                    ]
                else:
                    result = response.json()["data"][impact["result_key"]]

                result_list.append(result)

        cumulative_value = 0
        for r in result_list:
            if metric["result_key"] in r:
                cumulative_value += float(r[metric["result_key"]])

        if metric["operator"] == "divide":
            cumulative_value = cumulative_value / metric["denominator"]

        if metric["operator"] == "multiply":
            cumulative_value = cumulative_value * metric["denominator"]

        return cumulative_value

    else:
        response = requests.post(
            impact["api"],
            json={"query": impact["graphql"], "variables": impact["variables"]},
        )
        if response.status_code == 200:
            if impact["result_index"] is not None:
                result = response.json()["data"][impact["result_key"]][
                    impact["result_index"]
                ]
            else:
                result = response.json()["data"][impact["result_key"]]
                if isinstance(result, list):
                    result_list = result
                else:
                    result_list.append(result)

            value = 0
            for r in result_list:
                if metric["result_key"] in r:
                    value += float(r[metric["result_key"]])
            if metric["operator"] == "divide":
                value = value / metric["denominator"]
            if metric["operator"] == "multiply":
                value = value * metric["denominator"]
            return value


def refresh_regen(impact: dict, db_id: int):
    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # 1. Find the metric with the matching db_id
    metric = next((m for m in impact["metrics"] if m["db_id"] == db_id), None)
    if not metric:
        raise ValueError(f"No metric found for db_id {db_id}")

    # 2. Initialize accumulators
    denom_list = []
    hex_denom_list = []
    cumulative_retired_amount = 0
    bridged_amount = 0
    onchain_issued_amount = 0

    # 3. Fetch batches
    row = requests.post(
        impact["api"],
        headers={"Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": 176347957138,
            "method": "abci_query",
            "params": {"path": "/regen.ecocredit.v1.Query/Batches", "prove": False},
        },
    )

    value = row.json()["result"]["response"]["value"]
    decoded_bytes = base64.b64decode(value)

    message = QueryBatchesResponse()
    message.ParseFromString(decoded_bytes)

    for batch in message.batches:
        denom_list.append(batch.denom)

    # 4. Prepare hex encoded denoms
    for denom in denom_list:
        byte_denom = denom.encode("utf-8")
        length_hex = hex(len(denom))[2:].zfill(2)
        prefix = "0a" + length_hex
        hex_denom = prefix + binascii.hexlify(byte_denom).decode("utf-8")
        hex_denom_list.append({"hex": hex_denom, "string": denom})

    # 5. Query supplies
    for item in hex_denom_list:
        result = requests.post(
            impact["api"],
            headers={"Content-Type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": 717212259568,
                "method": "abci_query",
                "params": {
                    "path": "/regen.ecocredit.v1.Query/Supply",
                    "data": item["hex"],
                    "prove": False,
                },
            },
        )

        value = result.json()["result"]["response"]["value"]
        decoded_bytes = base64.b64decode(value)

        message = QuerySupplyResponse()
        message.ParseFromString(decoded_bytes)

        retired_amount = safe_float(message.retired_amount)
        tradable_amount = safe_float(message.tradable_amount)

        cumulative_retired_amount += retired_amount
        credit_class = item["string"].split("-")[0]

        if credit_class != "KSH01" and credit_class != "C03":
            bridged_amount += retired_amount + tradable_amount
        if credit_class == "KSH01" or credit_class == "USS01":
            onchain_issued_amount += retired_amount + tradable_amount

    # 6. Return only the value for the requested metric
    if metric["result_key"] == "cumulative_retired_amount":
        value = cumulative_retired_amount
    elif metric["result_key"] == "bridged_amount":
        value = bridged_amount
    elif metric["result_key"] == "onchain_issued_amount":
        value = onchain_issued_amount
    else:
        value = 0

    return round(value, 2)


def refresh_near(impact: dict, db_id: int):
    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # Find the metric with matching db_id
    metric = next((m for m in impact["metrics"] if m["db_id"] == db_id), None)
    if not metric:
        print(f"[DB ID {db_id}] No matching metric found in impact data")
        return 0.0

    # Prepare request body (if defined)
    post_body = json.loads(json.dumps(impact.get("body"))) if impact.get("body") else {}

    try:
        # Call Near API
        response = requests.post(
            impact["api"],
            headers={"Content-Type": "application/json"},
            json=post_body
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[DB ID {db_id}] Request error: {e}")
        return 0.0

    try:
        result = response.json()[impact["result_key"]][impact["result_index"]]
        decoded_response = "".join([chr(value) for value in result])
        data = json.loads(decoded_response)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"[DB ID {db_id}] Data parsing error: {e}")
        return 0.0

    # Calculate value for the matched metric
    cumulative_value = 0
    for item in data:
        value = safe_float(item.get(metric["result_key"]))
        if metric.get("denominator") is not None:
            try:
                value = value / int(metric["denominator"])
            except (ValueError, TypeError):
                print(f"[DB ID {db_id}] Invalid denominator: {metric['denominator']}")
        if metric.get("type") == "cumulative":
            cumulative_value += value

    return round(cumulative_value, 2)
