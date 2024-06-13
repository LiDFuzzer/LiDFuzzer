class InstanceGene():
    def __init__(self):
        self.position = None
        self.angle = None
        self.instance = None
        self.scale = None
        self.intensity = None
        self.asset = None

    def set_position(self, position):
        self.position = position

    def set_angle(self, angle):
        self.angle = angle

    def set_instance(self, instance):
        self.instance = instance

    def set_scale(self, scale):
        self.scale = scale

    def set_intensity(self, intensity):
        self.intensity = intensity

    def set_asset(self, asset):
        self.asset = asset
    
    def get_position(self):
        return self.position

    def get_angle(self):
        return self.angle

    def get_instance(self):
        return self.instance

    def get_scale(self):
        return self.scale

    def get_intensity(self):
        return self.intensity
    
    def get_asset(self):
        return self.asset
