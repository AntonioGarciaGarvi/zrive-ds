import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from jsonschema import validate, ValidationError
from typing import Dict, Any, List, Optional
from pandas import DataFrame

################################
### Constants declaration
################################
API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

START_DATE = "2010-01-01"
END_DATE = "2020-12-31"

meteo_api_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "generationtime_ms": {"type": "number"},
        "utc_offset_seconds": {"type": "integer"},
        "timezone": {"type": "string"},
        "timezone_abbreviation": {"type": "string"},
        "elevation": {"type": "number"},
        "daily_units": {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "temperature_2m_mean": {"type": "string"},
                "precipitation_sum": {"type": "string"},
                "wind_speed_10m_max": {"type": "string"},
            },
            "required": [
                "time",
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
            ],
        },
        "daily": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "array",
                    "items": {"type": "string", "format": "date"},
                },
                "temperature_2m_mean": {"type": "array", "items": {"type": "number"}},
                "precipitation_sum": {"type": "array", "items": {"type": "number"}},
                "wind_speed_10m_max": {"type": "array", "items": {"type": "number"}},
            },
            "required": [
                "time",
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
            ],
        },
    },
    "required": [
        "latitude",
        "longitude",
        "generationtime_ms",
        "utc_offset_seconds",
        "timezone",
        "timezone_abbreviation",
        "elevation",
        "daily_units",
        "daily",
    ],
}


################################
### API Functions
################################


def validate_api_response(response: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate an API response against a given JSON schema.

    Returns:
        bool: True if valid, False if invalid
    """

    try:
        validate(instance=response, schema=schema)
        print("API response is valid.")
        return True
    except ValidationError as err:
        print("API response is invalid:", err)
        return False


def call_api(api_url: str, cooloff: int, max_attempts: int) -> Optional[Any]:
    """
    Call an API endpoint with retry logic and exponential backoff.

    Args:
        api_url (str): The URL of the API endpoint to call.
        cooloff (int): Base number of seconds to wait before retrying.
        max_attempts (int): Maximum number of retry attempts after failures.

    Returns:
        Optional[Any]: Parsed JSON response if successful, otherwise None.

    Prints:
        Messages about rate limiting, request failures, and retries.
    """
    attempts = 0
    while attempts <= max_attempts:
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded. Cooling off...")
                delay = cooloff * (2**attempts)
                attempts += 1
                time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            delay = cooloff * (2**attempts)
            attempts += 1
            time.sleep(delay)

    return None


def get_data_meteo_api(
    city: str, start_date: str, end_date: str, variables: List[str]
) -> Dict[str, Any]:
    """
    Fetch meteorological data for a specific city and date range from the API.

    Args:
        city (str): Name of the city to fetch data for.
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.
        variables (List[str]): List of variable names to request (e.g., temperature, precipitation).

    Returns:
        Dict[str, Any]: The API response data parsed as a dictionary.

    Raises:
        ValidationError: If the API response does not match the expected schema.
    """
    api_meteo_url = API_URL
    latitude = COORDINATES[city]["latitude"]
    longitude = COORDINATES[city]["longitude"]
    variables_str = ",".join(variables)

    api_meteo_url += f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily={variables_str}"
    requested_data = call_api(api_url=api_meteo_url, cooloff=2, max_attempts=3)

    validate_api_response(requested_data, meteo_api_schema)

    return requested_data


###############################################
### Data processing and visualization functions
###############################################


def process_weather_data(df: DataFrame) -> DataFrame:
    """
    Process raw weather data to compute monthly mean and standard
    deviation for selected variables.

    Args:
        df (DataFrame): A DataFrame containing weather data with at least the columns
                        ['time', 'city', 'temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max'].

    Returns:
        DataFrame: A DataFrame with monthly mean and standard deviation of the selected variables,
                   indexed by city and time.
    """
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    # Only keep numeric columns for aggregation
    numeric_cols = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

    monthly_mean = (
        df.groupby("city")[numeric_cols]
        .resample("MS")  # MS = Month Start
        .mean()
        .reset_index()
    )
    monthly_std = df.groupby("city")[numeric_cols].resample("MS").std().reset_index()

    # Merge mean and std
    monthly_df = pd.merge(
        monthly_mean, monthly_std, on=["city", "time"], suffixes=("_mean", "_std")
    )

    return monthly_df


def plot_weather_data(monthly_df: DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot monthly weather data with error bars for multiple cities.

    Args:
        monthly_df (DataFrame): A DataFrame containing monthly mean and standard deviation
                                of weather variables with columns like 'temperature_2m_mean',
                                'precipitation_sum',
                                'wind_speed_10m_max', and their corresponding '_std' versions.
        save_path (Optional[str], optional): Path to save the plot as an image.
                                            If None, the plot is only displayed.

    Returns:
        None

    Displays:
        A multi-panel plot showing the temperature, precipitation, and wind speed trends
        with error bars.
    """
    variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
    titles = [
        "Monthly Average Temperature (°C)",
        "Monthly Total Precipitation (mm)",
        "Monthly Max Wind Speed (km/h)",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    for i, var in enumerate(variables):
        ax = axes[i]
        for city in monthly_df["city"].unique():
            city_data = monthly_df[monthly_df["city"] == city]

            mean_col = f"{var}_mean"
            std_col = f"{var}_std"

            # Plot mean with error bars
            ax.errorbar(
                city_data["time"],
                city_data[mean_col],
                yerr=city_data[std_col],
                label=city,
                fmt="-o",  # Line + circle markers
                capsize=4,  # Small caps at the ends of error bars
                alpha=0.8,
            )

        ax.set_title(titles[i], fontsize=14)
        ax.set_ylabel(titles[i])
        ax.grid(True)
        if i == 2:
            ax.set_xlabel("Date")
        else:
            ax.set_xlabel("")
        ax.legend(title="City")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    # plt.subplots_adjust(bottom=0.4)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def main() -> None:
    """
    Main execution function to obtain, process, and plot weather data.
    """

    weather_data_df = pd.DataFrame()
    for city_name, coordinates in COORDINATES.items():
        api_response = get_data_meteo_api(city_name, START_DATE, END_DATE, VARIABLES)
        if api_response:
            city_data = pd.DataFrame(api_response["daily"])
            city_data["city"] = city_name
            city_data["time"] = pd.to_datetime(city_data["time"])
            weather_data_df = pd.concat([weather_data_df, city_data], ignore_index=True)

    file_name = f"weather_data_frame_{START_DATE}_{END_DATE}.csv"
    weather_data_df.to_csv(file_name)

    monthly_df = process_weather_data(weather_data_df)
    plot_weather_data(monthly_df)


if __name__ == "__main__":
    main()
