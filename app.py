import requests

def get_bitcoin_price():
    # Define the URL for the CoinGecko API to fetch the current price of Bitcoin in US Dollars (USD)
    coingecko_api_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

    try:
        # Use the requests library to send a GET request to the API URL
        response = requests.get(coingecko_api_url)

        # Check if the request was successful (status code 200)
        # If it wasn't, this will raise an error that our except block will catch
        response.raise_for_status()

        # Convert the response data from JSON format into a Python dictionary
        data = response.json()

        # Extract the current price of Bitcoin in USD from the dictionary
        # The data looks like this: {'bitcoin': {'usd': 65000}}
        bitcoin_current_price = data["bitcoin"]["usd"]

        # Print the price in a clear, formatted string
        print(f"The current price of Bitcoin is: ${bitcoin_current_price}")

    except requests.exceptions.RequestException as error:
        # If any error occurs during the request (like no internet connection or the API is down),
        # print a user-friendly error message instead of a complex traceback
        print("Sorry, we couldn't fetch the current Bitcoin price at this time. Please check your internet connection or try again later.")

if __name__ == "__main__":
    # Call the function to get and print the Bitcoin price when the script is run directly
    get_bitcoin_price()
