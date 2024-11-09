import random

class DistributionArea:
    '''
    This class models the real-world scenario of a geographical/distribution area.
    '''
    def __init__(self, name: str, pop_density: int, proximity: int, infra: int, expansion_needed: bool):
        self.name = name
        self.pop_density = pop_density
        self.proximity = proximity
        self.infra = infra
        self.expansion_needed = expansion_needed

    def get_name(self) -> str:
        return self.name
    
    def get_density(self) -> int:
        return self.pop_density
    
    def get_proximity(self) -> int:
        return self.proximity
    
    def get_infra(self) -> int:
        return self.infra
    
    def get_expansion_needed(self) -> bool:
        return self.expansion_needed


def greedy_expansion_plan(areas: list[DistributionArea]) -> list:
    '''
    Drive code for deriving the expansion plan. 
    Returns a list of areas that requires expansion,
    sorted in order of density, proximity and infrastructure readiness. 
    '''
    # Sort areas based on population density, proximity to water sources, and infrastructure readiness
    areas.sort(key=lambda area: (area.get_density(), area.get_proximity(), area.get_infra()), reverse=True)
    expansion_plan = []
    
    for area in areas:
        if area.get_expansion_needed():
            expansion_plan.append(area.get_name())
    
    return expansion_plan



if __name__ == "__main__":
    num_areas = input("How many areas do you want to simulate?(int): ")
    num_areas = int(num_areas)


    areas = [
        DistributionArea(name=f"Area {i}", pop_density=random.randint(1, 10), infra=random.randint(1, 10), proximity=random.randint(1, 10), expansion_needed=random.choice([True, False]))
        for i in range(1, num_areas)
    ]


    plan = greedy_expansion_plan(areas)
    print("Expansion Plan:", plan)