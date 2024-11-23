'''
You are tasked with analyzing health insurance claims data represented in a JSON format. The data includes information about patients and their associated claims. You need to perform several tasks to extract insights and analyze patterns in the claims data.

Extract Related Claims
1: Write a function to extract all claims related to a given claim ID from the provided JSON data. Return a list of related claim IDs.

2: Using the same JSON data, write a function to find the maximum claim amount in the last 'n' days for a given patient ID.
'''


import json
from datetime import datetime, timedelta
from typing import List, Dict
# Test JSON Input
json_string = '{"patients": [{"id": "P001", "claims": [{"id": "C001", "amount": 100, "billingCode": "B001"}, {"id": "C002", "amount": 200, "billingCode": "B001"}, {"id": "C003", "amount": 150, "billingCode": "B002"}]}]}'
json_data = json.loads(json_string)

# extract all related claims to a given claim ID
def extract_related_claims(claim_id: str):
    '''
    1. get claim from given claim id
    2. get billing code of given claim
    3. extract claims with the same billing code
    '''
    claims_dict = dict(json_data) #redundant conversion
    patients = claims_dict['patients']
    # print(patients[0])
    # initialize variable to store the billing code of the give claim
    billing_code = None
    # loop over patients data to get the claim with the given id
    for p in patients:
        # p is a dict now
        for claim in p['claims']:
            # i get list of claims here
            if claim_id == claim['id']:
                billing_code = claim['billingCode']
                break

        # break the loop if claim is found
        if billing_code:
            break
    
    related_claims = []
    # loop over claims to get ones with same billing code as given claim
    for p in patients:
        for claim in p['claims']:
            if claim['billingCode'] == billing_code:
                related_claims.append(claim)

    # get the claim ids using list comprehension
    claim_ids = [c['id'] for c in related_claims]

    # return the related claim ids
    return claim_ids


# print(extract_related_claims('C001'))


def maximum_claim_amount(n: int, patient_id):
    patient = [] #json_data['patients']
