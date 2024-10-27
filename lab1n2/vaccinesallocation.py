import random

class Region:
    '''Class to model the Region object.'''
    def __init__(self, name, population, risk, infra):
        self.name = name
        self.population = population
        self.risk = risk
        self.infra = infra

def optimize_vaccine_distribution(regions, total_vaccines):
    n = len(regions)
    dp = [[0] * (total_vaccines + 1) for _ in range(n + 1)]
    decision = [[0] * (total_vaccines + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for v in range(1, total_vaccines + 1):
            # Do not allocate any vaccines to this region
            dp[i][v] = dp[i - 1][v]
            decision[i][v] = 0

            # Try allocating vaccines to this region
            for x in range(1, min(v, regions[i - 1].population) + 1):
                coverage = regions[i - 1].risk * regions[i - 1].infra
                potential_value = dp[i - 1][v - x] + coverage * x / regions[i - 1].population
                if potential_value > dp[i][v]:
                    dp[i][v] = potential_value
                    decision[i][v] = x

    # Trace back to find which regions received vaccines
    vaccines_left = total_vaccines
    supplied_regions = []

    for i in range(n, 0, -1):
        if decision[i][vaccines_left] > 0:
            vaccines_allocated = decision[i][vaccines_left]
            supplied_regions.append((regions[i - 1].name, vaccines_allocated))
            vaccines_left -= vaccines_allocated

    return dp[n][total_vaccines], supplied_regions

if __name__ == "__main__":
    num_of_regions = int(input("How many regions are there?: "))
    available_vaccines = int(input("How many vaccines are available?: "))

    regions = [
        Region(f"Region {i + 1}", population=random.randint(1000, 5000), risk=random.uniform(0.5, 1.0), infra=random.uniform(0.5, 1.0))
        for i in range(num_of_regions)
    ]

    optimized_result, supplied_regions = optimize_vaccine_distribution(regions, available_vaccines)

    print(f"Optimized vaccine distribution result: {optimized_result}")
    print("Supplied regions and their vaccines:")
    for region, vaccines in supplied_regions:
        print(f"{region}: {vaccines} vaccines")

