import os
import json

class Params():

    def __init__(self, json_path):
        self.json_path=json_path
        self.update(json_path)

    def save_as(self,json_path):
        with open(json_path,'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def save(self):
        with open(self.json_path,'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self,json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def merge_params(self, b):
        self.__dict__.update(b.__dict__)

    @property
    def dict(self):
        return self.__dict__
