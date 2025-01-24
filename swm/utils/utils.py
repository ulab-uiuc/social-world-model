from datetime import datetime
from typing import List, Dict


def unix_to_date(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M:%S')


def date_to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())

def filter_midnight_points(series: List[Dict[str, float]]) -> List[Dict[str, float]]:
    midnight_points = []
    for point in series:
        dt = datetime.fromtimestamp(point['t'])
        if dt.hour == 0 and dt.minute == 0:
            midnight_points.append(point)
    return midnight_points


def normalize_timestamp(ts: float) -> int:
    return ts - (ts % 60)
