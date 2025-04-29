from src.module_1.module_1_meteo_api import validate_api_response, meteo_api_schema


def test_validate_valid_response():
    valid_response = {
        "latitude": 40.416775,
        "longitude": -3.703790,
        "generationtime_ms": 1.23,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 667.0,
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_mean": "°C",
            "precipitation_sum": "mm",
            "wind_speed_10m_max": "km/h",
        },
        "daily": {
            "time": ["2010-01-01", "2010-01-02"],
            "temperature_2m_mean": [5.0, 6.0],
            "precipitation_sum": [0.0, 1.2],
            "wind_speed_10m_max": [10.5, 12.1],
        },
    }

    assert validate_api_response(valid_response, meteo_api_schema) is True


def test_validate_invalid_response():
    invalid_response = {
        "latitude": 40.416775,
        # 'longitude' is missing
        "generationtime_ms": 1.23,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 667.0,
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_mean": "°C",
            "precipitation_sum": "mm",
            "wind_speed_10m_max": "km/h",
        },
        "daily": {
            "time": ["2010-01-01"],
            "temperature_2m_mean": [5.0],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [10.5],
        },
    }

    assert validate_api_response(invalid_response, meteo_api_schema) is False
