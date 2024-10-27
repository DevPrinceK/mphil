import random

class Clinic:
    '''Class to model the clinic object in the question'''
    def __init__(self, name, demand, distance):
        self.name = name
        self.demand = demand
        self.distance = distance

def optimize_supply_chain(clinics: list[Clinic], total_supplies: int) -> tuple[int, list]:
    '''
    Helper funtion that uses a bottom-up dp approach to optimise supply chain.
    It takes in clinics (array/vector) and total_supplies (int/number)
    '''
    n = len(clinics)
    dp = [[0] * (total_supplies + 1) for _ in range(n + 1)]
    decision = [[0] * (total_supplies + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for s in range(1, total_supplies + 1):
            # Do not supply this clinic
            dp[i][s] = dp[i - 1][s]
            decision[i][s] = 0

            # Try supplying this clinic with part or all of its demand
            for x in range(1, min(s, clinics[i - 1].demand) + 1):
                potential_value = dp[i - 1][s - x] + (clinics[i - 1].distance * x / clinics[i - 1].demand)
                if potential_value > dp[i][s]:
                    dp[i][s] = potential_value
                    decision[i][s] = x

    # Trace back to find which clinics received supplies
    supplies_left = total_supplies
    supplied_clinics = []

    for i in range(n, 0, -1):
        if decision[i][supplies_left] > 0:
            supplies_allocated = decision[i][supplies_left]
            supplied_clinics.append((clinics[i - 1].name, supplies_allocated))
            supplies_left -= supplies_allocated

    return dp[n][total_supplies], supplied_clinics

if __name__ == "__main__":
    # let user define the figures
    print("PLEASE ENSURE TO ENTER VALID INTEGERS!")
    num_of_clinics = input("How many clinics are available?: ")
    num_of_clinics = int(num_of_clinics)

    available_supplies = input("How many supplies are available?: ")
    available_supplies = int(available_supplies)

    clinics = [
        Clinic(f"Clinic {i + 1}", demand=random.randint(100, 500), distance=random.randint(10, 30)) for i in range(num_of_clinics)
    ]

    optimized_result, supplied_clinics = optimize_supply_chain(clinics, available_supplies)

    print(f"Optimized supply chain result: {optimized_result}")
    print("Supplied clinics and their supplies:")
    for clinic, supplies in supplied_clinics:
        print(f"{clinic}: {supplies} supplies")

