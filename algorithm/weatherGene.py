class WeatherGene():
    def __init__(self):
        self.weather_type = None
        self.weather_para = None
        self.indice = None
        self.add_points = None
        self.add_intensity = None

    def set_weather_type(self, type):
        self.weather_type = type

    def set_weather_para(self, para):
        self.weather_para = para

    def set_indice(self, indice):
        self.indice = indice

    def set_add_points(self, add_points):
        self.add_points = add_points

    def set_add_intensity(self, add_intensity):
        self.add_intensity = add_intensity

    def get_weather_type(self):
        return self.weather_type

    def get_weather_para(self):
        return self.weather_para

    def get_indice(self):
        return self.indice
    
    def get_add_points(self):
        return self.add_points

    def get_add_intensity(self):
        return self.add_intensity

