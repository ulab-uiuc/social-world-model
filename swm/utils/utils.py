from datetime import datetime

def unix_to_date(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M:%S')

def date_to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())
