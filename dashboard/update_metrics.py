import requests, base64, binascii, json, os, time, logging
from functools import wraps
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from dune_client.client import DuneClient

from .regen_pb2 import QueryBatchesResponse, QuerySupplyResponse
from .utils import get_nested_value

logger = logging.getLogger(__name__)

class RefreshMetricResponse(BaseModel):
    db_id: int = Field(..., description="ID of the metric in both DB and Impact JSON")
    value: float | None = Field(None, description="Refreshed value or None if failed")
    single: bool = Field(False, description="Whether the metric is single value (i.e., not cumulative)")

def safe_refresh(max_retries=2, delay=2.0):
    """
    Decorator to wrap refresh functions with retry logic and error handling.
    Logs errors and returns None instead of raising.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    value = func(*args, **kwargs)

                    # Validation: allow int, float, dict, list, or any object with __dict__
                    if value is None:
                        raise ValueError("Returned None")

                    if not isinstance(value, (int, float, list, dict)) and not hasattr(value, "__dict__"):
                        raise ValueError(f"Invalid value returned: {type(value)}")

                    return value

                except Exception as e:
                    logger.warning(
                        f"[{func.__name__}] attempt {attempt} failed: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{func.__name__}] failed after {max_retries} attempts: {e}"
                        )
                        return None
        return wrapper
    return decorator

@safe_refresh(max_retries=3, delay=3)
def refresh_dune(impact: dict) -> list[RefreshMetricResponse]:
    # Dune initialization
    dune = DuneClient(os.getenv("DUNE_KEY"))
    results = []

    for metric in impact["metrics"]:
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

        result = RefreshMetricResponse(
            db_id=metric["db_id"],
            value=value
        )
        results.append(result)

    return results

@safe_refresh(max_retries=3, delay=3)
def refresh_client(impact: dict) -> list[RefreshMetricResponse]:
    results = []

    if impact["method"] == "POST":
        post_body = json.loads(json.dumps(impact["body"]))
        if "start_date" in impact:
            current_date = datetime.today()
            difference = current_date - datetime.strptime(impact["start_date"]["date"], "%Y-%m-%d")
            week_number = (difference.days // 7 + impact["start_date"]["week"]) - 1
            post_body["week_number"] = int(week_number)

        response = requests.post(impact["api"], json=post_body)
        if impact["result_index"] is not None and impact["result_key"] is not None:
            metric_data = response.json()[impact["result_key"]][impact["result_index"]]
        else:
            metric_data = response.json()

        list_value = 0

        for metric in impact["metrics"]:
            if "list_name" in metric:
                for i in metric_data[metric["list_name"]]:
                    list_value += float(i[metric["result_key"]])
                value = list_value
            else:
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

            result = RefreshMetricResponse(
                db_id=metric["db_id"],
                value=formatted_value,
                single=True if "single" in metric else False
            )

            results.append(result)

        return results

    elif impact["method"] == "GET":
        if impact["result_key"] is not None:
            response = requests.get(impact["api"])

            for metric in impact["metrics"]:
                value_path = impact["result_key"] + "." + metric["result_key"]
                value = round(float(get_nested_value(response.json(), value_path)), 2)

                if metric["denominator"] is not None:
                    value = value / int(metric["denominator"])

                result = RefreshMetricResponse(
                    db_id=metric["db_id"],
                    value=value
                )

                results.append(result)

            return results
        else:
            list_value = 0

            for metric in impact["metrics"]:
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
                    
                
                result = RefreshMetricResponse(
                    db_id=metric["db_id"],
                    value=round(value,2)
                )

                results.append(result)

            return results

@safe_refresh(max_retries=3, delay=3)
def refresh_subgraph(impact: dict) -> list[RefreshMetricResponse]:
    cumulative_value = 0
    results = []

    for metric in impact["metrics"]:
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
        
        result = RefreshMetricResponse(
            db_id=metric["db_id"],
            value=cumulative_value
        )
        results.append(result)

    return results

@safe_refresh(max_retries=3, delay=3)
def refresh_vebetter(impact: dict) -> list[RefreshMetricResponse]:
    vebetter_api = "https://graph.vet/subgraphs/name/vebetter/dao"
    results = []

    response = requests.post(
            vebetter_api,
            json={"query": impact["graphql"], "variables": impact["variables"]},
        )
    
    if response.status_code == 200:
        data = response.json()['data']['statsAppSustainabilities'][0]
        value = 0

        for metric in impact["metrics"]:
            if metric['result_key'] in data:
                if metric['result_index'] is not None:
                    value = float(data[metric['result_key']][metric['result_index']])
                else:
                    value = float(data[metric['result_key']])
            else:
                return None
            
            if metric['operator'] == "divide":
                value = value / metric['denominator']

            if metric['operator'] == "multiply":
                value = value * metric['denominator']
            
            result = RefreshMetricResponse(
                db_id=metric["db_id"],
                value=value
            )
            results.append(result)

    return results

@safe_refresh(max_retries=3, delay=3)
def refresh_graphql(impact: dict) -> list[RefreshMetricResponse]:
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
        results = []

        for metric in impact["metrics"]:
            for r in result_list:
                if metric["result_key"] in r:
                    cumulative_value += float(r[metric["result_key"]])

            if metric["operator"] == "divide":
                cumulative_value = cumulative_value / metric["denominator"]

            if metric["operator"] == "multiply":
                cumulative_value = cumulative_value * metric["denominator"]
            
            result = RefreshMetricResponse(
                db_id=metric["db_id"],
                value=cumulative_value
            )
            results.append(result)

        return results

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
            results = []

            for metric in impact["metrics"]:
                for r in result_list:
                    if metric["result_key"] in r:
                        value += float(r[metric["result_key"]])
                if metric["operator"] == "divide":
                    value = value / metric["denominator"]
                if metric["operator"] == "multiply":
                    value = value * metric["denominator"]
                
                result = RefreshMetricResponse(
                    db_id=metric["db_id"],
                    value=value
                )
                results.append(result)
                
            return results

@safe_refresh(max_retries=3, delay=3)
def refresh_regen(impact: dict) -> list[RefreshMetricResponse]:
    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

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
    results = []
    for metric in impact["metrics"]:
        if metric["result_key"] == "cumulative_retired_amount":
            value = cumulative_retired_amount
        elif metric["result_key"] == "bridged_amount":
            value = bridged_amount
        elif metric["result_key"] == "onchain_issued_amount":
            value = onchain_issued_amount
        else:
            value = 0
        
        result = RefreshMetricResponse(
            db_id=metric["db_id"],
            value=round(value,2)
        )

        results.append(result)

    return results

@safe_refresh(max_retries=3, delay=3)
def refresh_near(impact: dict) -> list[RefreshMetricResponse]:
    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
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
        return 0.0

    try:
        result = response.json()[impact["result_key"]][impact["result_index"]]
        decoded_response = "".join([chr(value) for value in result])
        data = json.loads(decoded_response)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        return 0.0

    # Calculate value for the matched metric
    cumulative_value = 0
    results = []
    for metric in impact["metrics"]:
        for item in data:
            value = safe_float(item.get(metric["result_key"]))
            if metric.get("denominator") is not None:
                try:
                    value = value / int(metric["denominator"])
                except (ValueError, TypeError):
                    value = 0.0
            if metric.get("type") == "cumulative":
                cumulative_value += value
        
        result = RefreshMetricResponse(
            db_id=metric["db_id"],
            value=round(cumulative_value,2)
        )
        results.append(result)

    return results
