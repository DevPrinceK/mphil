'''
Here, I am going to use classes to model the real-life objects 
involved in the problem we're trying to solve
[Objects]: Region**, Vaccine.
'''
import random


class Region:
    '''
    This class models the Region object.
    [Params]: name, population, risk level, infrastructure
    '''
    def __init__(self, name: str, population: int, rist: str, infra: str):
        self.name = name
        self.population = population
        self.risk = rist
        self.infra = infra

    def getName(self) -> str:
        return self.name
    

    def setName(self, name: str) -> None:
        self.name = name


    def getPopulation(self, population: str) -> str:
        self.population = population
        return self.population
    

    def setPopulation(self, population: int) -> None:
        self.population = population
    

    def getRiskLevel(self) -> str:
        return self.risk
    

    def setRiskLevel(self, risk: int) -> None:
        self.risk = risk
    

regions: list[Region] = [Region(f"Region {1}", random.randint(1000, 5000), random.randint(1, 10), random.randint(1, 10))]