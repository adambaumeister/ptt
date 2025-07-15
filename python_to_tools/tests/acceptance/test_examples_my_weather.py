
def test_get_location():
    from python_to_tools.examples.my_weather import get_user_location, get_weather
    location = get_user_location()
    result = get_weather(location.lat, location.lon)
    print(result)