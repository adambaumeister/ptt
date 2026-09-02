
def test_get_location():
    from examples.my_weather import get_user_location, get_weather
    location = get_user_location()
    result = get_weather(location.lat, location.lon)
    print(result)