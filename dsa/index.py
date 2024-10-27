# from dsa.models import Region
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


    def getPopulation(self) -> str:
        return self.population
    

    def setPopulation(self, population: int) -> None:
        self.population = population
    

    def getRiskLevel(self) -> str:
        return self.risk
    

    def setRiskLevel(self, risk: int) -> None:
        self.risk = risk
    

    def getInfrastructure(self) -> str:
        return self.infra
    

    def setRiskLevel(self, infra: int) -> None:
        self.infra = infra
    


def distribute_vaccine(available_vaccines: int, regions: list[Region]):
    '''A DP algorithm for vaccine allocation'''
    n = len(regions)
    dp = [[0] * (n + 1) for _ in range(available_vaccines + 1)]
    
    for v in range(1, available_vaccines + 1):
        for r in range(1, n + 1):
            # Do not allocate any vaccine to region r
            dp[v][r] = dp[v][r - 1]
            
            # etxract region r
            population = regions[r - 1].getPopulation()
            risk_level = regions[r - 1].getRiskLevel()
            infrastructure = regions[r - 1].getInfrastructure()
            region_name = regions[r - 1].getName()
            
            # Calculate the effective coverage factor
            coverage_factor = risk_level * infrastructure
            
            # Calculate the number of vaccines to allocate
            vaccines_to_allocate = min(v, population)
            
            # Update dp table
            dp[v][r] = max(dp[v][r], dp[v - vaccines_to_allocate][r - 1] + coverage_factor * vaccines_to_allocate)

    
    return dp[available_vaccines][n]




if __name__ == "__main__":
    available_vaccines = int(input("How many vaccines are available (Please Enter a valid number): "))
    regions = int(input("How many regions need vaccines: "))
    regions: list[Region] = [
        Region(f'Region {i}', random.randint(100, 5000), random.randint(1, 10), random.randint(1, 10)) for i in range(1, regions + 1)
    ]
    max_coverage = distribute_vaccine(available_vaccines, regions)

    print(f"Maximum coverage with {available_vaccines} vaccines: {max_coverage}")