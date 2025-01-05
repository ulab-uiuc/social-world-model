import os
import time
import datetime
import matplotlib.pyplot as plt

from py_clob_client.client import ClobClient
from dotenv import load_dotenv

# Load the environment variables from the specified .env file

def main():
    host = "https://clob.polymarket.com"
    key = os.getenv("PK")
    chain_id = 137  # Polygon Mainnet chain ID
    
    # Ensure the private key is loaded correctly
    if not key:
        raise ValueError("Private key not found. Please set PK in the environment variables.")
    
    # Initialize the client with your private key
    client = ClobClient(host, key=key, chain_id=chain_id)
    
    # Create or derive API credentials (this is where the API key, secret, and passphrase are generated)
    try:
        api_creds = client.create_or_derive_api_creds()
        print("API Key:", api_creds.api_key)
        print("Secret:", api_creds.api_secret)
        print("Passphrase:", api_creds.api_passphrase)
        # You should now save these securely (e.g., store them in your .env file)
    except Exception as e:
        print("Error creating or deriving API credentials:", e)
    
    # Retrieve price history (last 3 days, fidelity=60 means one point per hour)
    start_ts = int(time.time()) - 10 * 24 * 60 * 60
    end_ts   = int(time.time())
    price_data = client.get_price_history_for_interval(
        #token_id="54774190602932495681624596813895280541057113822779394733336244613748178149294",
        #token_id="46226686344811071373865769241015888474371014887184710566323582300308375002290",
        #token_id="28182404005967940652495463228537840901055649726248190462854914416579180110833",
        token_id="26978210066243057557403011639180032346872416391650353082320087028857924937495",
        interval="max",
        fidelity=60,
    )
    
    # Print raw response for debugging (optional)
    print(price_data)

    # Extract timestamps and prices from the response
    timestamps = [item['t'] for item in price_data['history']]
    prices = [item['p'] for item in price_data['history']]

    # Convert Unix timestamps to datetime objects
    datetimes = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

    # Create a Matplotlib figure
    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, prices, marker='o', linestyle='-', color='b')  # basic line plot

    # Label axes and set title
    plt.xlabel('Date/Time')
    plt.ylabel('Price')
    plt.title('Price History (Last 3 Days)')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
