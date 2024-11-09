
import random


class Zone:
    '''
    This class models the real-world scenario of a geographical area
    in need of water.
    '''
    def __init__(self, name: str, priority: int, region: int, s_availability: int, water_needed: int):
        self.name = name
        self.priority = priority
        self.region = region
        self.s_availability = s_availability
        self.water_needed = water_needed

    def get_name(self):
        return self.name
    
    def get_priority(self):
        return self.name
    
    def get_region(self):
        return self.region
    
    def get_availability(self):
        return self.s_availability
    
    def get_water_needed(self):
        return self.water_needed



def greedy_allocation(total_water, zones: list[Zone]) -> dict:
    '''
    Driver code for allocating water to various zones
    taking into account their levels of need, priorities, etc...
    It returns a dictionary of zones and amount of water they need vrs allocated.
    '''
    # Sort zones based on crop priority, region, and seasonal availability
    zones.sort(key=lambda zone: (zone.get_priority(), zone.get_region(), zone.get_availability()), reverse=True)
    
    allocation_plan = {}
    
    for zone in zones:
        if total_water > 0:
            water_needed = zone.get_water_needed()
            water_allocated = min(total_water, water_needed)
            allocation_plan[zone.get_name()] = {
                "Water Needed": zone.get_water_needed(),
                "Water Allocated": water_allocated,
            }
            total_water -= water_allocated
    
    return allocation_plan


if __name__ == "__main__":

    num_zones = input("How many zones do you want to simulate?(int): ")
    num_zones = int(num_zones)

    total_water = input("How much water is available?(int): ")
    total_water = int(total_water)


    zones = [
        Zone(name=f"Area {i}",  priority=random.randint(1, 5), region=random.randint(1, 10), s_availability=random.randint(1, 10), water_needed=random.randint(10, 25) )
        for i in range(1, num_zones)
    ]

    plan = greedy_allocation(total_water=total_water, zones=zones)
    print("Allocation Plan:", plan)