import requests

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import datetime

class AuthManager:
    def __init__(self, key_id: str, key_file_path: str):
        self.key_id = key_id
        self.private_key = self._load_private_key(key_file_path)

    def _load_private_key(self, file_path: str):
        with open(file_path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )

    def generate_headers(self, method: str, path: str) -> dict:
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        msg_string = str(timestamp) + method + path
        signature = self._sign_message(msg_string)
        
        return {
            'Content-Type': 'application/json',
            'KALSHI-ACCESS-KEY': self.key_id,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': str(timestamp)
        }

    def _sign_message(self, message: str) -> str:
        msg_bytes = message.encode('utf-8')
        signature = self.private_key.sign(
            msg_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

# should be 00:00:00 of the day
now = datetime.datetime.now()
start_ts = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()) - 86400 * 450
end_ts = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()) - 86400 * 448
start_ts = 1728324000 - 864000* 200
end_ts = 1732060800 - 864000* 100
print(start_ts, end_ts)
url = f"https://api.elections.kalshi.com/trade-api/v2/series/TIPPINGPOINT/events/TIPPINGPOINT-24/forecast_percentile_history?percentiles=5000&start_ts={start_ts}&end_ts={end_ts}&period_interval=1440"

#url = "https://api.elections.kalshi.com/trade-api/v2/series/TIPPINGPOINT/events/TIPPINGPOINT-24/forecast_percentile_history?percentiles=5000&start_ts=1728306000&end_ts=1736190240&&period_interval=1440"



auth_manager = AuthManager(key_id="f9ad5959-eeac-4de1-bfa6-03b675c8da73", key_file_path="./swm.txt")
headers = auth_manager.generate_headers("GET", url)


response = requests.get(url, headers=headers)

print(response.json())